# -----------------------------
# utils.py
# -----------------------------
import os
import sys
from pathlib import Path
import torch
import tqdm
import numpy as np
from sacrebleu.metrics import BLEU
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer
import time
from pathlib import Path



cousin_dir = Path(__file__).resolve().parents[1]

def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Train a ByteLevelBPETokenizer from the 'examples', then convert to a HF AutoTokenizer.
    """
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        [[ex[src_key], ex[tgt_key]] for ex in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>']
    )
    tokenizer.save(f"{workdir}/tokenizer.json")
    # We assume some config is in workdir (like config.json) from GPT2
    # Then transform to HF tokenizer
    t = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None
    )
    return t

def evaluate_bleu(examples, gen_sents, tgt_key):
    bleu_obj = BLEU()
    score = bleu_obj.corpus_score(
        hypotheses=gen_sents,
        references=[[ex[tgt_key] for ex in examples]]
    )
    return {'bleu': score.score}

def save_grad_weights(model, rank):
    """Saves gradients to ./tests/model{rank}_gradients.pth no matter what."""
    # Create the "./tests" folder if it doesn't exist (relative to current working dir):
    tests_path = Path("./tests")
    tests_path.mkdir(exist_ok=True)

    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().cpu()

    # Force saving under ./tests
    save_path = tests_path / f"model{rank}_gradients.pth"
    torch.save(gradients, save_path)
    print(f"[Rank {rank}] Gradients saved to: {save_path.resolve()}")


def collate_batch(examples, src_key, tgt_key, tokenizer, model_max_length, device):
    token_ids, tgt_mask = [], []
    pad_id = tokenizer.vocab['<pad>']
    max_length = model_max_length + 1

    for ex in examples:
        src_toks = tokenizer(f"{ex[src_key]}<eos_{src_key}>")['input_ids']
        tgt_toks = tokenizer(f"{ex[tgt_key]}<eos_{tgt_key}>")['input_ids']
        combined = src_toks + tgt_toks
        mask = [0]*len(src_toks) + [1]*len(tgt_toks)

        combined = combined[:max_length]
        mask = mask[:max_length]
        # pad if needed
        pad_count = max_length - len(combined)
        combined += [pad_id]*pad_count
        mask += [0]*pad_count

        token_ids.append(combined)
        tgt_mask.append(mask)

    token_ids = torch.tensor(token_ids, device=device)
    tgt_mask = torch.tensor(tgt_mask, device=device)

    return {
        'input_ids': token_ids[:, :-1],
        'labels': token_ids[:, 1:],
        'label_token_weights': tgt_mask[:, 1:]
    }

def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    model.eval()
    losses = []
    for batch in (prog_bar := tqdm.tqdm(examples, desc=f"Evaluating {desc}")):
        with torch.no_grad():
            loss = loss_fn(model, batch)
        losses.append(loss.item())
        prog_bar.set_postfix(loss=f"{loss.item():.3f}")
    return float(np.mean(losses))

def generate(model, examples, src_key, tgt_key, tokenizer, model_max_length, device, desc):
    """
    Generate from model until <eos_{tgt_key}> or hits max_length.
    """
    model.eval()
    gen_sents = []

    for ex in tqdm.tqdm(examples, desc=f"Generating {desc}"):
        token_ids = tokenizer(f"{ex[src_key]}<eos_{src_key}>")['input_ids']
        len_src = len(token_ids)

        while len(token_ids) <= model_max_length:
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([token_ids], device=device)).logits
                next_tok = torch.argmax(logits[0, -1]).item()

            # If next_tok is the <eos_{tgt_key}>, break
            if next_tok == tokenizer.vocab.get(f'<eos_{tgt_key}>', None):
                break
            token_ids.append(next_tok)

        # decode from len_src onward
        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents

def loss_fn(model, batch):
    logits = model(input_ids=batch['input_ids']).logits
    loss_raw = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch['labels'].reshape(-1),
        reduction='none'
    )
    # Weighted only on "target" portion
    weights = batch['label_token_weights'].reshape(-1)
    return torch.sum(loss_raw * weights) / torch.sum(weights)

def train(model, optimizer, examples, batch_size, collate_fn, desc, rank=0, average_gradients_fn=None):
    model.train()
    tokens_per_sec = []
    tokens_num = []

    for batch in (prog_bar := tqdm.tqdm(examples, desc=f"Training {desc}")):
        t0 = time.time()

        optimizer.zero_grad()
        l = loss_fn(model, batch)
        l.backward()

        if average_gradients_fn is not None:
            average_gradients_fn(model)

        optimizer.step()

        btime = time.time() - t0
        tokens = np.prod(batch['input_ids'].shape)
        tokens_per_sec.append(tokens / btime)
        tokens_num.append(tokens)

        prog_bar.set_postfix(loss=f"{l.item():.3f}", tokens_per_sec=f"{tokens/btime:.1f}")

    return float(np.mean(tokens_per_sec)), tokens_num

