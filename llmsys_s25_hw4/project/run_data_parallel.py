# -----------------------------
# run_data_parallel.py
# -----------------------------
import sys
from pathlib import Path

cousin_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(cousin_dir))

import time
import os
import argparse
import json
import datasets
import numpy as np
from transformers import AutoConfig, GPT2LMHeadModel

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

from functools import partial
from data_parallel.dataset import partition_dataset
from utils import (
    get_tokenizer,
    evaluate_bleu,
    save_grad_weights,
    collate_batch,
    evaluate_loss,
    generate,
    train
)

PYTEST = False

# ASSIGNMENT 4.1
def average_gradients(model):
    """
    Aggregate the gradients from different processes/GPUs:
      1. all_reduce to sum
      2. divide by the world_size for average
    """
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def setup(rank, world_size, backend):
    """
    1) MASTER_ADDR='127.0.0.1'
    2) MASTER_PORT='11868'
    3) init_process_group
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '11868'  # If port is busy, pick another
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def run_dp(
    rank,
    world_size,
    backend,
    dataset_name='bbaaaa/iwslt14-de-en-preprocess',
    model_max_length=128,
    n_epochs=10,
    batch_size=128,
    learning_rate=1e-4
):
    # 1) Init distributed
    setup(rank, world_size, backend)

    workdir = './workdir'
    os.makedirs(workdir, exist_ok=True)

    # 2) Create GPT2 config & model
    config = AutoConfig.from_pretrained('gpt2')
    config.save_pretrained(workdir)

    model = GPT2LMHeadModel(config).to(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 3) Load dataset
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    # For demonstration, limit data
    dataset['train'] = dataset['train'][:5000]
    dataset['validation'] = dataset['validation'][:1000]
    dataset['test'] = dataset['test'][:100]

    # 4) Tokenizer
    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config.vocab_size,
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir
    )

    # 5) Collation & partition
    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        device=rank
    )
    train_loader = partition_dataset(
        rank=rank,
        world_size=world_size,
        dataset=dataset['train'],
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset['validation'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset['test'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    total_time = []
    total_tokens_per_sec = []

    # 6) Main training
    for epoch_idx in range(n_epochs):
        desc = f"rank {rank}/{world_size} epoch {epoch_idx}/{n_epochs}"
        start = time.time()

        ############################################
        # If we are in PYTEST mode, do just 1 batch
        # (like your minimal snippet).
        ############################################
        if PYTEST:
            print('##### PYTEST IN')
            model.train()
            one_batch = next(iter(train_loader), None)
            if one_batch is not None:
                optimizer.zero_grad()
                logits = model(input_ids=one_batch['input_ids']).logits

                # Instead of `.view(...)`, do `.reshape(...)`
                vocab_size = logits.shape[-1]
                logits_2d = logits.reshape(-1, vocab_size)
                labels_1d = one_batch['labels'].reshape(-1)

                loss = torch.nn.functional.cross_entropy(
                    logits_2d,
                    labels_1d
                )
                loss.backward()
                average_gradients(model)
                optimizer.step()

                print('#### BEFORE SAVE')
                save_grad_weights(model, rank)
                print(f"[Rank {rank}] Pytest mode: saved grad file => break out.")
            break  # Done after 1 batch
        else:
            print('##### NOT PYTEST ')            
            # Normal training for a full epoch
            avg_tokens_per_sec, _ = train(
                model=model,
                optimizer=optimizer,
                examples=train_loader,
                batch_size=batch_size,
                collate_fn=collate_fn,
                desc=desc,
                rank=rank,
                average_gradients_fn=average_gradients
            )

            end = time.time()
            train_time = end - start
            total_time.append(train_time)
            total_tokens_per_sec.append(avg_tokens_per_sec)

            print(f"Epoch {epoch_idx}, Rank {rank}: train_time={train_time:.2f}s, tokens/s={avg_tokens_per_sec:.1f}")

            # Evaluate
            val_loss = evaluate_loss(
                model=model,
                examples=val_loader,
                batch_size=batch_size,
                collate_fn=collate_fn,
                desc=desc
            )
            print(f"Epoch {epoch_idx}, Rank {rank}: val_loss={val_loss:.3f}")

            # Generate
            gen_sents = generate(
                model=model,
                examples=dataset['test'],
                src_key=src_key,
                tgt_key=tgt_key,
                tokenizer=tokenizer,
                model_max_length=model_max_length,
                device=rank,
                desc=desc
            )
            gen_examples = []
            for example, gen_sent in zip(dataset['test'], gen_sents):
                gen_examples.append({'example': example, 'gen': gen_sent})
            json.dump(gen_examples, open(f"{workdir}/rank{rank}_gen_epoch{epoch_idx}.json", "w"), indent=4)

            eval_scores = evaluate_bleu(
                examples=dataset['test'],
                gen_sents=gen_sents,
                tgt_key=tgt_key
            )
            print(f"Epoch {epoch_idx}, Rank {rank} => {eval_scores}")

            json.dump(
                {
                    'validation_loss': val_loss,
                    **eval_scores,
                    'train_time': train_time,
                    'tokens_per_sec': avg_tokens_per_sec
                },
                open(f"{workdir}/rank{rank}_results_epoch{epoch_idx}.json", 'w')
            )

    # Summaries if not in pytest
    if not PYTEST and len(total_time) > 0:
        print(f"Rank {rank} => total_time avg={np.mean(total_time):.2f}, std={np.std(total_time):.2f}, "
              f"tokens/s avg={np.mean(total_tokens_per_sec):.1f}, std={np.std(total_tokens_per_sec):.1f}")

    dist.destroy_process_group()


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pytest', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='bbaaaa/iwslt14-de-en-preprocess')
    parser.add_argument('--model_max_length', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    PYTEST = args.pytest

    world_size = args.world_size
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_dp,
            args=(
                rank,
                world_size,
                backend,
                args.dataset,
                args.model_max_length,
                args.n_epochs,
                args.batch_size,
                args.learning_rate
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# import sys
# from pathlib import Path

# cousin_dir = Path(__file__).resolve().parents[1]
# sys.path.append(str(cousin_dir))

# from functools import partial
# import time
# import os
# import argparse
# import tqdm
# import json
# import datasets
# import numpy as np
# from transformers import AutoConfig, GPT2LMHeadModel
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader 
# import torch.distributed as dist
# import torch.multiprocessing as mp  # Note: changed to mp so we can set spawn start method.

# from data_parallel.dataset import partition_dataset
# from utils import get_tokenizer, evaluate_bleu, save_grad_weights, collate_batch, evaluate_loss, generate, train

# PYTEST = False

# # ASSIGNMENT 4.1
# def average_gradients(model):
#     '''Aggregate the gradients from different GPUs
    
#     1. Iterate through the parameters of the model.
#     2. Use torch.distributed.all_reduce to sum the gradients across all processes.
#     3. Average the gradients by dividing by the world_size.
#     '''
#     world_size = dist.get_world_size()  # how many processes total
#     for param in model.parameters():
#         if param.grad is not None:
#             dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
#             param.grad /= world_size


# # ASSIGNMENT 4.1
# def setup(rank, world_size, backend):
#     '''
#     Setup Process Group

#     1. Set environment variables MASTER_ADDR as '127.0.0.1'
#        and MASTER_PORT as '11868'
#     2. Use torch.distributed to init the process group
#     '''
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '11868'
#     dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


# def run_dp(
#     rank, world_size, backend,
#     dataset_name='bbaaaa/iwslt14-de-en-preprocess',
#     model_max_length=128,
#     n_epochs=10,
#     batch_size=128,
#     learning_rate=1e-4):
#     """
#     The main function each process will run.
#     It sets up distributed training, loads data, runs train/eval,
#     and cleans up the process group.
#     """
#     # 1) Initialize the process group
#     setup(rank, world_size, backend)

#     workdir = f'./workdir'
#     os.makedirs(workdir, exist_ok=True)

#     config = AutoConfig.from_pretrained('gpt2')
#     config.save_pretrained(workdir)
    
#     # Move model to current rank (device)
#     model = GPT2LMHeadModel(config=config).to(rank)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#     # 2) Load dataset
#     dataset = {
#         split: datasets.load_dataset(dataset_name, split=split)['translation']
#         for split in ['train', 'validation', 'test']
#     }
#     src_key, tgt_key = 'de', 'en'

#     # (Make smaller for demonstration)
#     dataset['train'] = dataset['train'][:5000]
#     dataset['validation'] = dataset['validation'][:1000]
#     dataset['test'] = dataset['test'][:100]

#     # 3) Build tokenizer
#     tokenizer = get_tokenizer(
#         examples=dataset['train'],
#         vocab_size=config.vocab_size,
#         src_key=src_key,
#         tgt_key=tgt_key,
#         workdir=workdir)

#     collate_fn = partial(
#         collate_batch,
#         src_key=src_key,
#         tgt_key=tgt_key,
#         tokenizer=tokenizer,
#         model_max_length=model_max_length,
#         device=rank)

#     # 4) Partition dataset for current rank
#     train_loader = partition_dataset(
#         rank,
#         world_size,
#         dataset['train'],
#         batch_size=batch_size,
#         collate_fn=collate_fn)

#     val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#     total_time = []
#     total_tokens_per_sec = []

#     # 5) Main training/evaluation loop
#     for epoch_idx in range(n_epochs):
#         desc = f'rank {rank}/{world_size} epoch {epoch_idx}/{n_epochs}'

#         start = time.time()

#         # The train function calls loss.backward() then average_gradients after backward
#         avg_tokens_per_sec, _ = train(
#             model=model,
#             optimizer=optimizer,
#             examples=train_loader,
#             batch_size=batch_size,
#             collate_fn=collate_fn,
#             desc=desc,
#             rank=rank,
#             average_gradients_fn=average_gradients
#         )

#         end = time.time()
#         training_time = end - start

#         # When not in pytest mode, print stats and run validation
#         if not PYTEST:
#             print(f'Epoch {epoch_idx} on Rank {rank}: Training Time = {training_time}, Tokens_per_sec = {avg_tokens_per_sec}')
#             total_time.append(training_time)
#             total_tokens_per_sec.append(avg_tokens_per_sec)

#             validation_loss = evaluate_loss(
#                 model=model,
#                 examples=val_loader,
#                 batch_size=batch_size,
#                 collate_fn=collate_fn,
#                 desc=desc)
#             print(f'Epoch {epoch_idx} on Rank {rank}: Validation Loss = {validation_loss}')

#             gen_sents = generate(
#                 model=model,
#                 examples=dataset['test'],
#                 src_key=src_key,
#                 tgt_key=tgt_key,
#                 tokenizer=tokenizer,
#                 model_max_length=model_max_length,
#                 device=rank,
#                 desc=desc)

#             gen_examples = []
#             for example, gen_sent in zip(dataset['test'], gen_sents):
#                 gen_examples.append({'example': example, 'gen': gen_sent})
#             json.dump(gen_examples, open(f'{workdir}/rank{rank}_gen_epoch{epoch_idx}.json', 'w'), indent=4)

#             eval_scores = evaluate_bleu(
#                 examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
#             print(f'Epoch {epoch_idx} on Rank {rank}: {eval_scores}')

#             json.dump(
#                 {
#                     'validation_loss': validation_loss,
#                     **eval_scores,
#                     'training_time': training_time,
#                     'tokens_per_sec': avg_tokens_per_sec
#                 },
#                 open(f'{workdir}/rank{rank}_results_epoch{epoch_idx}.json', 'w')
#             )
#         else:
#             # If in pytest mode, just save the gradient from one batch & break
#             save_grad_weights(model, rank)
#             break

#     # Print summary for each rank
#     if not PYTEST and len(total_time) > 0:
#         print(f'Rank {rank} training time: avg:{np.mean(total_time)}, std:{np.std(total_time)}, '
#               f'tokens_per_sec: avg:{np.mean(total_tokens_per_sec)}, std:{np.std(total_tokens_per_sec)}')

#     # Cleanup distributed resources
#     dist.destroy_process_group()


# if __name__ == '__main__':
#     # -----------------------------------------
#     # Set spawn mode (needed for CUDA + fork)
#     # -----------------------------------------
#     mp.set_start_method("spawn", force=True)

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pytest', type=bool, default=False)
#     parser.add_argument('--dataset', type=str, default='bbaaaa/iwslt14-de-en-preprocess')
#     parser.add_argument('--model_max_length', type=int, default=128)
#     parser.add_argument('--n_epochs', type=int, default=10)
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--learning_rate', type=float, default=1e-4)
#     parser.add_argument('--world_size', type=int, default=2)
#     args = parser.parse_args()

#     PYTEST = args.pytest

#     processes = []
#     world_size = args.world_size
#     backend = 'nccl' if torch.cuda.is_available() else 'gloo'

#     # Launch each process
#     for rank in range(world_size):
#         p = mp.Process(
#             target=run_dp,
#             args=(rank, world_size, backend,
#                   args.dataset,
#                   args.model_max_length,
#                   args.n_epochs,
#                   args.batch_size,
#                   args.learning_rate)
#         )
#         p.start()
#         processes.append(p)

#     # Wait for processes to finish
#     for p in processes:
#         p.join()

