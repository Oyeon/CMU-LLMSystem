import os
import json
import fire
import time
import tqdm
import random
import contextlib
import datasets
import numpy as np

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
    Obtrain IWSLT (de-en) dataset.
    """
    # dataset = {
    #     split: datasets.load_dataset(dataset_name, split=split)['translation']
    #     for split in ['train', 'validation', 'test']
    # }
    dataset = {
        split: datasets.load_dataset(dataset_name, use_auth_token=True, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }    
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(
                example[tgt_key].split()) < model_max_length
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
    Trains a tokenizer on the provided dataset examples and saves the tokenizer configuration.
    """
    tokenizer = ByteLevelBPETokenizer()
    # We expect "examples" to be a list of dict: {'de': "...", 'en': "..."}
    # We'll train from pairs of [src_text, tgt_text].
    tokenizer.train_from_iterator(
        [[ex[src_key], ex[tgt_key]] for ex in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>']
    )

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    # Load with transformers' AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None
    )
    return tokenizer

def collate_batch(examples, src_key, tgt_key, tokenizer, model_max_length, backend):
    """
    Prepares a batch for training:
      - For each example, tokenize DE+<eos_de> then EN+<eos_en>,
      - Concatenate, pad to length model_max_length,
      - "input_ids" => tokens[:-1],
      - "labels" => tokens[1:],
      - "label_token_weights" => 0 for source tokens, 1 for target.
    """
    token_ids, tgt_token_mask = [], []
    pad_token_id = tokenizer.vocab['<pad>']

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

        token_ids.append(combined)
        tgt_token_mask.append(mask)

    token_ids = np.array(token_ids)
    tgt_token_mask = np.array(tgt_token_mask)

    input_ids = token_ids[:, :-1]
    labels = token_ids[:, 1:]
    label_token_weights = tgt_token_mask[:, 1:]

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
    The MLE loss for next-token prediction, ignoring source tokens.
    """
    idx = batch['input_ids']
    idx.requires_grad_(True)

    logits = model(idx=idx)  # [batch, seq_len, vocab_size]

    bs, seqlen, vocab_size = logits.shape
    logits = logits.view(bs*seqlen, vocab_size)

    targets = batch['labels'].view(bs*seqlen)
    mask = batch['label_token_weights'].view(bs*seqlen)

    targets.requires_grad_(True)

    # cross-entropy on all positions
    ce = minitorch.nn.softmax_loss(logits=logits, target=targets)

    # but we only average over target tokens
    return ((ce * mask).sum() / mask.sum())

def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
    """
    One epoch of training:
      - Shuffle examples, take n_samples,
      - for each batch, do forward + backward + step
    """
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]

    for i in (prog_bar := tqdm.trange(0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch = collate_fn(examples=examples[i:i+batch_size])
        t0 = time.time()

        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)

        t1 = time.time()
        loss.backward()
        t2 = time.time()

        optimizer.step()
        t3 = time.time()

        # Some debug prints
        print(f"Forward: {t1 - t0:.3f} s")
        print(f"Backward: {t2 - t1:.3f} s")
        print(f"Opt.step: {t3 - t2:.3f} s")

        batch_time = time.time() - t0
        n_tokens = np.prod(batch['input_ids'].shape)
        prog_bar.set_postfix(
            loss=f"{loss.item():.4f}",
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
        batch = collate_fn(examples=examples[i:i+batch_size])
        loss = loss_fn(batch=batch, model=model)
        losses.append(loss.item())
        prog_bar.set_postfix(loss=f"{loss.item():.4f}")
    return float(np.mean(losses))


def gather_row(tensor2d, row_idx):
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
             desc):
    """
    Greedy argmax decoding with minimal indexing:
      1) Flatten from [1, seq_len, vocab_size] to [seq_len, vocab_size].
      2) gather_row(...) to extract last_logits as 1D.
    """
    model.eval()
    gen_sents = []

    with no_grad():
        for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
            token_ids = tokenizer(f"{example[src_key]}<eos_{src_key}>")['input_ids']
            len_src = len(token_ids)

            while len(token_ids) < model_max_length:
                seq_len = len(token_ids)

                idx = minitorch.tensor(token_ids, backend=backend).view(1, seq_len)
                logits = model(idx=idx)  # => [1, seq_len, vocab_size]

                # Flatten => [seq_len, vocab_size]
                _, s_len, vocab_size = logits.shape
                flattened = logits.view(s_len, vocab_size)

                # Gather last row => [vocab_size]
                last_logits = gather_row(flattened, seq_len - 1)

                # Convert to numpy, then argmax
                vals = last_logits.to_numpy()  # shape [vocab_size]
                next_id = int(np.argmax(vals))

                if next_id == tokenizer.vocab[f"<eos_{tgt_key}>"]:
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
    n_epochs=20, #20
    batch_size=128,
    learning_rate=0.02,
    samples_per_epoch=20000, #20000
    n_vocab=10000,
    n_embd=256,
    seed=11111
):
    """
    Example usage:
      python project/run_machine_translation.py \
        --dataset_name=iwslt2017 \
        --model_max_length=40 \
        --n_epochs=1
    """
    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    backend = minitorch.TensorBackend(CudaKernelOps)

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
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    # load dataset, filter by max_length
    dataset, src_key, tgt_key = get_dataset(dataset_name, model_max_length)

    # train BPE tokenizer
    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config['n_vocab'],
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir
    )

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend
    )

    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'
        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc
        )

        val_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc
        )
        print(f"Epoch {epoch_idx}: Validation Loss = {val_loss:.4f}")

        # Generate from test set
        gen_sents = generate(
            model=model,
            examples=dataset['test'],
            src_key=src_key,
            tgt_key=tgt_key,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
            backend=backend,
            desc=desc
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
