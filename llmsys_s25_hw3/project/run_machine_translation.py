import os
import json
import fire
import time
import tqdm
import random
import contextlib
import datasets
import numpy as np
import math  # for math.isnan, math.isinf

from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer
from functools import partial

import minitorch
from minitorch.modules_transformer import DecoderLM
from minitorch.cuda_kernel_ops import CudaKernelOps

@contextlib.contextmanager
def no_grad():
    """Dummy no_grad for minitorch; does nothing special."""
    yield

def get_dataset(dataset_name, model_max_length):
    """
    Obtain IWSLT (de-en) dataset.
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, use_auth_token=True, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    # Filter out examples that exceed the token count
    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(example[tgt_key].split()) < model_max_length
        ]
        for split in dataset.keys()
    }

    # for quick tests, limit the test set
    dataset['test'] = dataset['test'][:100]

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Trains a ByteLevelBPETokenizer on the provided dataset examples 
    and saves the tokenizer config. Then loads with AutoTokenizer.
    """
    tokenizer = ByteLevelBPETokenizer()
    # We expect "examples" to be a list of dict: {'de': "...", 'en': "..."}
    tokenizer.train_from_iterator(
        [[ex[src_key], ex[tgt_key]] for ex in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>', '<unk>']
    )

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    # Load with transformers' AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token='<unk>'  # Ensure <unk> is recognized
    )
    return tokenizer


def collate_batch(examples, src_key, tgt_key, tokenizer, model_max_length, backend, n_vocab):
    """
    Prepares a batch:
      - For each example, tokenize DE+<eos_de> and EN+<eos_en>
      - Concatenate => [source + target], pad up to model_max_length
      - 'mask' = [0 for source] + [1 for target]
      - If mask.sum()==0 => skip that example
      - Return a dict of Tensors: {input_ids, labels, label_token_weights}
    """
    pad_token_id = tokenizer.vocab['<pad>']
    unk_token_id = tokenizer.vocab['<unk>']

    valid_input_ids = []
    valid_labels = []
    valid_label_weights = []

    for example in examples:
        de_text = f"{example[src_key]}<eos_{src_key}>"
        en_text = f"{example[tgt_key]}<eos_{tgt_key}>"

        token_ids_src = tokenizer(de_text)['input_ids']
        token_ids_tgt = tokenizer(en_text)['input_ids']

        combined = token_ids_src + token_ids_tgt
        mask = [0]*len(token_ids_src) + [1]*len(token_ids_tgt)

        # Truncate + pad
        combined = combined[:model_max_length]
        mask = mask[:model_max_length]

        pad_needed = model_max_length - len(combined)
        combined += [pad_token_id]*pad_needed
        mask += [0]*pad_needed

        # Replace out-of-range tokens with <unk>
        for i, tid in enumerate(combined):
            if tid >= n_vocab:
                combined[i] = unk_token_id

        # Now separate input_ids / labels
        input_ids_ = combined[:-1]
        labels_ = combined[1:]
        label_mask_ = mask[1:]  # same shape as labels_

        # If target portion is entirely truncated => sum(label_mask_)=0 => skip
        if sum(label_mask_) == 0:
            continue

        valid_input_ids.append(input_ids_)
        valid_labels.append(labels_)
        valid_label_weights.append(label_mask_)

    if len(valid_input_ids) == 0:
        return None

    input_ids = np.array(valid_input_ids, dtype=np.int64)
    labels = np.array(valid_labels, dtype=np.int64)
    label_token_weights = np.array(valid_label_weights, dtype=np.float32)

    input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
    labels = minitorch.tensor_from_numpy(labels, backend=backend)
    label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_token_weights': label_token_weights
    }


def loss_fn(batch, model):
    """
    The MLE loss for next-token prediction, ignoring source tokens (mask=0).
    """
    idx = batch['input_ids']
    idx.requires_grad_(True)

    # forward => [batch, seq_len, vocab_size]
    logits = model(idx=idx)

    # [디버깅1] logits 체크
    logits_np = logits.to_numpy()
    # if np.isnan(logits_np).any() or np.isinf(logits_np).any():
    #     print(f"[DEBUG] logits has NaN/Inf! shape={logits.shape}, min={logits_np.min()}, max={logits_np.max()}")

    bs, seqlen, vocab_size = logits.shape
    logits = logits.view(bs*seqlen, vocab_size)

    targets = batch['labels'].view(bs*seqlen)
    mask = batch['label_token_weights'].view(bs*seqlen)
    targets.requires_grad_(True)

    # compute cross-entropy
    ce = minitorch.nn.softmax_loss(logits=logits, target=targets)

    # [디버깅2] ce 체크
    ce_np = ce.to_numpy()
    # if np.isnan(ce_np).any() or np.isinf(ce_np).any():
    #     print(f"[DEBUG] cross-entropy has NaN/Inf! shape={ce_np.shape}, min={ce_np.min()}, max={ce_np.max()}")

    denom = mask.sum()
    if denom == 0:
        return ce.mean()
    return ((ce * mask).sum() / denom)


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
    """
    One epoch of training:
      - Shuffle examples, take n_samples,
      - for each batch => forward + backward + step
    """
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]

    for i in (prog_bar := tqdm.trange(0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch_data = collate_fn(examples=examples[i:i+batch_size])
        if batch_data is None:
            continue

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch_data, model=model)

        # [디버깅3] loss.item() NaN/Inf 체크
        loss_val = loss.item()
        # if math.isnan(loss_val) or math.isinf(loss_val):
            # print(f"[DEBUG] Found NaN/Inf in loss => {loss_val} at step {i}. Stopping training.")
            # break  # or continue, or import pdb; pdb.set_trace()

        t1 = time.time()
        loss.backward()

        # (선택) 파라미터 grad 검사
        # for param in model.parameters():
        #     if param.grad is not None:
        #         g_arr = param.grad.to_numpy()
        #         if np.isnan(g_arr).any() or np.isinf(g_arr).any():
        #             print(f"[DEBUG] Found NaN/Inf in grad of {param.name} at step {i}.")
        #             # import pdb; pdb.set_trace()

        t2 = time.time()
        optimizer.step()
        t3 = time.time()

        # Debug prints
        print(f"Forward: {t1 - t0:.3f} s")
        print(f"Backward: {t2 - t1:.3f} s")
        print(f"Opt.step: {t3 - t2:.3f} s")

        batch_time = time.time() - t0
        n_tokens = np.prod(batch_data['input_ids'].shape)
        prog_bar.set_postfix(
            loss=f"{loss_val:.4f}",
            tokens_per_sec=f"{n_tokens/batch_time:.1f}",
            lr=optimizer.lr
        )


def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    """
    Evaluates average cross-entropy on the given dataset.
    """
    model.eval()
    losses = []
    for i in (prog_bar := tqdm.trange(0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
        batch_data = collate_fn(examples=examples[i:i+batch_size])
        if batch_data is None:
            continue

        loss = loss_fn(batch=batch_data, model=model)
        losses.append(loss.item())
        prog_bar.set_postfix(loss=f"{loss.item():.4f}")

    if len(losses) == 0:
        return float('inf')
    return float(np.mean(losses))


def gather_row(tensor2d, row_idx):
    """
    Helper function for generation: extract 1 row as a 1D tensor.
    """
    S, V = tensor2d.shape
    row_values = []
    for col in range(V):
        val = tensor2d[row_idx, col]
        row_values.append(val)
    return minitorch.tensor(row_values, backend=tensor2d.backend)


def generate(model,
             examples,
             src_key,
             tgt_key,
             tokenizer,
             model_max_length,
             backend,
             desc,
             n_vocab):
    """
    Greedy decoding:
      - If next token >= n_vocab => force <unk>.
      - Stop if we see <eos_{tgt_key}> or we hit model_max_length.
    """
    model.eval()
    gen_sents = []
    unk_token_id = tokenizer.vocab['<unk>']
    eos_tgt_id = tokenizer.vocab.get(f'<eos_{tgt_key}>', -1)

    with no_grad():
        for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
            token_ids = tokenizer(f"{example[src_key]}<eos_{src_key}>")['input_ids']
            len_src = len(token_ids)

            # clamp out-of-range to <unk>
            for i in range(len(token_ids)):
                if token_ids[i] >= n_vocab:
                    token_ids[i] = unk_token_id

            while len(token_ids) < model_max_length:
                seq_len = len(token_ids)
                idx = minitorch.tensor(token_ids, backend=backend).view(1, seq_len)
                logits = model(idx=idx)  # => [1, seq_len, vocab_size]

                _, s_len, vocab_size = logits.shape
                flattened = logits.view(s_len, vocab_size)

                last_logits = gather_row(flattened, seq_len - 1)
                vals = last_logits.to_numpy()  # shape [vocab_size]
                next_id = int(np.argmax(vals))

                # clamp out-of-range
                if next_id >= n_vocab:
                    next_id = unk_token_id

                # stop if <eos_{tgt_key}>
                if next_id == eos_tgt_id:
                    break
                token_ids.append(next_id)

            gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents


def evaluate_bleu(examples, gen_sents, tgt_key):
    """
    Evaluate BLEU score of 'gen_sents' vs. references in examples.
    """
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[ex[tgt_key] for ex in examples]]
        ).score
    }


def main(
    dataset_name='bbaaaa/iwslt14-de-en-preprocess',
    model_max_length=40,
    n_epochs=1,
    batch_size=8,
    learning_rate=0.02,        # Must match command-line arg name
    samples_per_epoch=20000,
    n_vocab=10000,
    n_embd=256,
    seed=11111,
    use_fused_kernel=False,
):
    print(f"[INFO] use_fused_kernel = {use_fused_kernel}")
    print(f"[INFO] learning_rate = {learning_rate}")

    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    backend = minitorch.TensorBackend(CudaKernelOps)

    # 1) Load dataset
    dataset, src_key, tgt_key = get_dataset(dataset_name, model_max_length)

    # 2) Train BPE tokenizer
    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=n_vocab,  # initial target
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir
    )

    real_vocab_size = tokenizer.vocab_size
    print(f"Actual tokenizer vocab_size = {real_vocab_size}")
    if real_vocab_size > n_vocab:
        print(f"WARNING: real vocab size {real_vocab_size} > config n_vocab={n_vocab} -> updating model n_vocab")
        n_vocab = real_vocab_size

    # 3) Build model
    config = {
        'n_vocab': n_vocab,
        'n_embd': n_embd,
        'n_head': 8,
        'n_positions': model_max_length,
        'p_dropout': 0.1,
        'ln_eps': 1e-5,
        'backend': backend
    }
    model = DecoderLM(**config)

    # 4) Setup optimizer
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    # 5) Prepare collate_fn
    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend,
        n_vocab=n_vocab
    )

    # 6) Training & Evaluation
    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        # Train
        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc
        )

        # Validation
        val_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc
        )
        print(f"Epoch {epoch_idx}: Validation Loss = {val_loss:.4f}")

        # Generate from test
        gen_sents = generate(
            model=model,
            examples=dataset['test'],
            src_key=src_key,
            tgt_key=tgt_key,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
            backend=backend,
            desc=desc,
            n_vocab=n_vocab
        )

        # Save generations
        gen_examples = []
        for example, gen_sent in zip(dataset['test'], gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        with open(f"{workdir}/gen_epoch{epoch_idx}.json", "w") as f:
            json.dump(gen_examples, f, indent=4)

        # Evaluate BLEU
        eval_scores = evaluate_bleu(dataset['test'], gen_sents, tgt_key)
        print(f"Epoch {epoch_idx}: {eval_scores}")

        # Save eval results
        with open(f"{workdir}/eval_results_epoch{epoch_idx}.json", "w") as f:
            json.dump({'validation_loss': float(val_loss), **eval_scores}, f)

if __name__ == "__main__":
    fire.Fire(main)


# from functools import partial
# import time
# import os
# import fire
# import tqdm
# import json
# import random
# import datasets
# import numpy as np
# from sacrebleu.metrics import BLEU
# from transformers import AutoTokenizer
# from tokenizers import ByteLevelBPETokenizer
# import contextlib


# import minitorch
# from minitorch.modules_transformer import DecoderLM

# backend_name = "CudaKernelOps"
# if backend_name == "CudaKernelOps":
#     from minitorch.cuda_kernel_ops import CudaKernelOps
#     BACKEND = minitorch.TensorBackend(CudaKernelOps)

    

# def get_dataset(dataset_name, model_max_length):
#     """
#     Obtrain IWSLT (de-en) dataset.
#     """
#     # dataset = {
#     #     split: datasets.load_dataset(dataset_name, split=split)['translation']
#     #     for split in ['train', 'validation', 'test']
#     # }
#     dataset = {
#         split: datasets.load_dataset(dataset_name, use_auth_token=True, split=split)['translation']
#         for split in ['train', 'validation', 'test']
#     }    
#     src_key, tgt_key = 'de', 'en'

#     dataset = {
#         split: [
#             example for example in dataset[split]
#             if len(example[src_key].split()) + len(
#                 example[tgt_key].split()) < model_max_length
#         ]
#         for split in dataset.keys()
#     }

#     # for quick tests, limit the test set
#     dataset['test'] = dataset['test'][:100]

#     print(json.dumps(
#         {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
#         indent=4))

#     return dataset, src_key, tgt_key


# def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
#     """
#     Trains a tokenizer on the provided dataset examples and saves the tokenizer configuration.
#     """
#     tokenizer = ByteLevelBPETokenizer()
#     tokenizer.train_from_iterator(
#         [[ex[src_key], ex[tgt_key]] for ex in examples],
#         vocab_size=vocab_size,
#         special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

#     tokenizer.save(f'{workdir}/tokenizer.json')
#     json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

#     tokenizer = AutoTokenizer.from_pretrained(
#         workdir,
#         eos_token=None,
#         bos_token=None,
#         pad_token=None,
#         unk_token=None)

#     return tokenizer


# def collate_batch(
#         examples, src_key, tgt_key, tokenizer, model_max_length, backend):
#     """
#     Prepares a batch of examples for model training or evaluation by tokenizing and padding them.
#     """
#     token_ids, tgt_token_mask = [], []
#     pad_token_id = tokenizer.vocab['<pad>']

#     for example in examples:
#         token_ids_src = tokenizer(f"{example[src_key]}<eos_{src_key}>")['input_ids']
#         token_ids_tgt = tokenizer(f"{example[tgt_key]}<eos_{tgt_key}>")['input_ids']

#         # Concatenate source and target, then build target-mask
#         example_token_ids = token_ids_src + token_ids_tgt
#         example_tgt_token_mask = (
#             [0] * len(token_ids_src) + [1] * len(token_ids_tgt)
#         )

#         # Truncate to max_length, then pad
#         example_token_ids = example_token_ids[:model_max_length]
#         example_tgt_token_mask = example_tgt_token_mask[:model_max_length]
#         pad_needed = model_max_length - len(example_token_ids)

#         example_token_ids += [pad_token_id] * pad_needed
#         example_tgt_token_mask += [0] * pad_needed

#         token_ids.append(example_token_ids)
#         tgt_token_mask.append(example_tgt_token_mask)

#     token_ids = np.array(token_ids)
#     tgt_token_mask = np.array(tgt_token_mask)

#     # Shift inputs vs. labels by one position
#     input_ids = token_ids[:, :-1]
#     labels = token_ids[:, 1:]
#     label_token_weights = tgt_token_mask[:, 1:]

#     input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
#     labels = minitorch.tensor_from_numpy(labels, backend=backend)
#     label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)

#     return {
#         'input_ids': input_ids,
#         'labels': labels,
#         'label_token_weights': label_token_weights
#     }


# def loss_fn(batch, model):
#     """
#     The MLE loss for a batch (softmax cross-entropy over next tokens).
#     """
#     idx = batch['input_ids']
#     idx.requires_grad_(True)

#     # forward => logits shape [batch_size, seq_len, vocab_size]
#     logits = model(idx=idx)

#     bs, l, c = logits.shape
#     logits = logits.view(bs * l, c)

#     targets = batch['labels'].view(bs * l)
#     label_token_weights = batch['label_token_weights'].view(bs * l)
#     targets.requires_grad_(True)

#     # standard cross-entropy (logits -> softmax -> pick target)
#     loss = minitorch.nn.softmax_loss(logits=logits, target=targets)

#     # average over *only* the target tokens
#     return ((loss * label_token_weights).sum() / label_token_weights.sum())


# def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
#     """
#     Single epoch: shuffle examples, take n_samples, run training loop on mini-batches.
#     """
#     model.train()
#     random.shuffle(examples)
#     examples = examples[:n_samples]

#     for i in (prog_bar := tqdm.trange(0, len(examples), batch_size, desc=f'Training ({desc})')):
#         batch = collate_fn(examples=examples[i:i + batch_size])

#         t0 = time.time()
#         optimizer.zero_grad()
#         loss = loss_fn(batch=batch, model=model)
#         t1 = time.time()

#         loss.backward()
#         t2 = time.time()

#         optimizer.step()
#         t3 = time.time()

#         # Some debug prints (optional)
#         print(f"Forward: {t1 - t0:.3f} s")
#         print(f"Backward: {t2 - t1:.3f} s")
#         print(f"Opt.step: {t3 - t2:.3f} s")

#         batch_time = time.time() - t0
#         n_tokens = np.prod(batch['input_ids'].shape)
#         prog_bar.set_postfix(
#             tokens_per_sec=n_tokens / batch_time,
#             loss=f"{loss.item():.4f}",
#             lr=optimizer.lr)


# def evaluate_loss(model, examples, batch_size, collate_fn, desc):
#     """
#     Computes average loss over the entire `examples` set.
#     """
#     model.eval()
#     losses = []

#     for i in (prog_bar := tqdm.trange(0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
#         batch = collate_fn(examples=examples[i:i + batch_size])
#         loss = loss_fn(batch=batch, model=model)
#         losses.append(loss.item())
#         prog_bar.set_postfix(loss=f"{loss.item():.4f}")

#     return np.mean(losses)

# @contextlib.contextmanager
# def no_grad():
#     """No-op for minitorch. Does nothing."""
#     yield


# def generate(model,
#              examples,
#              src_key,
#              tgt_key,
#              tokenizer,
#              model_max_length,
#              backend,
#              desc):
#     """
#     Greedy argmax decoding, example by example:
#       1) Encode the source + <eos_de>.
#       2) Loop until <eos_en> or max_length.
#       3) Argmax the last position’s logits to pick next token.
#       4) Decode the portion after the source => final translation.
#     """
#     model.eval()
#     gen_sents = []

    
#     with no_grad():  # no gradient tracking for generation
#         for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
#             # 1) tokenize the German source + <eos_de>
#             token_ids = tokenizer(f"{example[src_key]}<eos_{src_key}>")['input_ids']
#             len_src = len(token_ids)

#             # 2) repeatedly predict next token
#             while len(token_ids) < model_max_length:
#                 # shape => [1, current_seq_length]
#                 idx = minitorch.tensor(token_ids, backend=backend).view(1, -1)
#                 logits = model(idx=idx)          # => shape [1, seq_len, vocab_size]

#                 # get the last position's logits => shape [vocab_size]
#                 last_logits = logits[0, -1, :]

#                 # argmax => next token
#                 next_id = int(last_logits.argmax().item())
#                 if next_id == tokenizer.vocab[f"<eos_{tgt_key}>"]:
#                     break
#                 token_ids.append(next_id)

#             # 3) decode from beyond the source => predicted English
#             gen_sents.append(tokenizer.decode(token_ids[len_src:]))

#     return gen_sents


# def evaluate_bleu(examples, gen_sents, tgt_key):
#     """
#     Evaluate BLEU for the generated translations vs. reference.
#     """
#     return {
#         'bleu': BLEU().corpus_score(
#             hypotheses=gen_sents,
#             references=[[ex[tgt_key] for ex in examples]]
#         ).score
#     }


# def main(dataset_name='bbaaaa/iwslt14-de-en-preprocess',
#          model_max_length=40,
#          n_epochs=20,
#          batch_size=128,
#          learning_rate=0.02,
#          samples_per_epoch=20000,
#          n_vocab=10000,
#          n_embd=256,
#          seed=11111):
#     """
#     The main function to train and evaluate the model on a specified dataset.
#     """
#     np.random.seed(seed)
#     random.seed(seed)

#     workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
#     os.makedirs(workdir, exist_ok=True)

#     backend = minitorch.TensorBackend(CudaKernelOps)

#     config = {
#         'n_vocab': n_vocab,
#         'n_embd': n_embd,
#         'n_head': 8,
#         'n_positions': model_max_length,
#         'p_dropout': 0.1,
#         'ln_eps': 1e-5,
#         'backend': backend
#     }

#     model = DecoderLM(**config)
#     optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

#     dataset, src_key, tgt_key = get_dataset(
#         dataset_name=dataset_name,
#         model_max_length=model_max_length
#     )

#     tokenizer = get_tokenizer(
#         examples=dataset['train'],
#         vocab_size=config['n_vocab'],
#         src_key=src_key,
#         tgt_key=tgt_key,
#         workdir=workdir
#     )

#     collate_fn = partial(
#         collate_batch,
#         src_key=src_key,
#         tgt_key=tgt_key,
#         tokenizer=tokenizer,
#         model_max_length=model_max_length,
#         backend=backend
#     )

#     # Main training loop
#     for epoch_idx in range(n_epochs):
#         desc = f'epoch {epoch_idx} / {n_epochs}'

#         train(
#             model=model,
#             optimizer=optimizer,
#             examples=dataset['train'],
#             n_samples=samples_per_epoch,
#             batch_size=batch_size,
#             collate_fn=collate_fn,
#             desc=desc
#         )

#         validation_loss = evaluate_loss(
#             model=model,
#             examples=dataset['validation'],
#             batch_size=batch_size,
#             collate_fn=collate_fn,
#             desc=desc
#         )
#         print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss:.4f}')

#         gen_sents = generate(
#             model=model,
#             examples=dataset['test'],
#             src_key=src_key,
#             tgt_key=tgt_key,
#             tokenizer=tokenizer,
#             model_max_length=model_max_length,
#             backend=backend,
#             desc=desc
#         )

#         # Save generations to JSON
#         gen_examples = []
#         for example, gen_sent in zip(dataset['test'], gen_sents):
#             gen_examples.append({'example': example, 'gen': gen_sent})
#         json.dump(
#             gen_examples,
#             open(f'{workdir}/gen_epoch{epoch_idx}.json', 'w'),
#             indent=4
#         )

#         eval_scores = evaluate_bleu(
#             examples=dataset['test'],
#             gen_sents=gen_sents,
#             tgt_key=tgt_key
#         )
#         print(f'Epoch {epoch_idx}: {eval_scores}')

#         json.dump(
#             {'validation_loss': float(validation_loss), **eval_scores},
#             open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w')
#         )


# if __name__ == '__main__':
#     fire.Fire(main)
