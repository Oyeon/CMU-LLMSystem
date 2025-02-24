
import random
import pdb

import embeddings

import sys
sys.path.append('../')
import minitorch

from datasets import load_dataset

backend_name = "CudaKernelOps"
if backend_name == "CudaKernelOps":
    from minitorch.cuda_kernel_ops import CudaKernelOps
    BACKEND = minitorch.TensorBackend(CudaKernelOps)

BATCH = 10

def RParam(*shape):
    """
    Create a random parameter with the given shape,
    scaled to a smaller range for stable training.
    """
    # Example: 0.1 * (rand - 0.5) => random in [-0.05, 0.05]
    r = 0.1 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


###############################################################
# Linear Layer
###############################################################
class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        # BEGIN ASSIGN1_3
        # 1. Initialize self.weights as a random parameter of (in_size, out_size)
        # 2. Initialize self.bias as a random parameter of (out_size,)
        # 3. Store out_size in self.out_size
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size
        # END ASSIGN1_3

    def forward(self, x):
        # x shape: (batch, in_size)
        # print("DEBUG [Linear] input x shape =", x.shape)
        batch, in_size = x.shape  # <-- Will fail if x is 3D    
        batch, in_size = x.shape
        
        # BEGIN ASSIGN1_3
        # 1. x is already (batch, in_size) – so no additional reshape needed, unless you want to use x.view(...)
        # 2. Reshape self.weights to (in_size, out_size)
        w_reshaped = self.weights.value.view(in_size, self.out_size)
        
        # 3. Matrix multiply => (batch, out_size)
        out = x @ w_reshaped
        
        # 4. Add bias, broadcast to (batch, out_size)
        out = out + self.bias.value.view(1, self.out_size)
        
        # Return result (shape: (batch, out_size))
        return out
        # END ASSIGN1_3


###############################################################
# Network
###############################################################
class Network(minitorch.Module):
    """
    Implement a MLP for SST-2 sentence sentiment classification.

    The steps are:
    1. Average over the sentence length. (from shape [B, L, E] -> [B, E])
    2. Apply a Linear layer to hidden_dim, then ReLU, then Dropout.
    3. Apply a second Linear to size 1 (for 1 output class).
    4. Apply a Sigmoid activation (final shape [batch]).
    """

    def __init__(
        self,
        embedding_dim=50,
        hidden_dim=32,
        dropout_prob=0.5,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob

        # BEGIN ASSIGN1_3
        # 1. Construct two linear layers:
        #    - First:  Linear(embedding_dim, hidden_dim)
        #    - Second: Linear(hidden_dim, 1)
        self.linear1 = Linear(embedding_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, 1)
        # END ASSIGN1_3

    def forward(self, embeddings):
        """
        embeddings shape: [batch, sentence_length, embedding_dim]
        """
        # BEGIN ASSIGN1_3
        # 1. Average over sentence length => shape [batch, embedding_dim]
        # print("DEBUG [Network] input embeddings shape =", embeddings.shape)
        # 1) Manual sum + divide for mean
        x = embeddings.sum(dim=1)  # shape -> likely (5, 1, 150)

        # 2) Divide by sentence length => shape still (5, 1, 150)
        sentence_length = embeddings.shape[1]  # e.g. 15
        x = x / sentence_length

        # 3) Reshape from (5, 1, 150) to (5, 150)
        x = x.view(x.shape[0], x.shape[2])        
        # print("DEBUG [Network] after mean shape =", x.shape)        
        # 2. Apply the first linear layer
        x = self.linear1(x)

        # 3. Apply ReLU and dropout
        x = x.relu()
        x = minitorch.dropout(x, rate=self.dropout_prob, ignore=not self.training)

        # 4. Apply the second linear layer => shape [batch, 1]
        x = self.linear2(x)

        # 5. Apply sigmoid => shape still [batch, 1], then reshape to [batch]
        x = x.sigmoid()
        x = x.view(x.shape[0])

        return x
        # END ASSIGN1_3


# ===================================================================
# Evaluation Helper Methods
# ===================================================================
def get_predictions_array(y_true, model_output):
    predictions_array = []
    model_output = model_output.view(model_output.shape[0])
    
    for j in range(model_output.shape[0]):
        true_label = y_true[j]
        logit = model_output[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array):
    correct = 0
    for (y_true, y_pred, logit) in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array) if len(predictions_array) > 0 else 0.0


best_val = 0.0

def default_log_fn(
    epoch,
    train_loss,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    global best_val
    # Track best validation accuracy
    if validation_accuracy:
        best_val = max(best_val, validation_accuracy[-1])

    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")


# ===================================================================
# Sentiment Trainer
# ===================================================================
class SentenceSentimentTrain:
    """
    The trainer class of sentence sentiment classification.
    """

    def __init__(self):
        self.model = Network()  # Your MLP from Problem 3

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=BATCH,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)

        # Create an Adam optimizer
        optim = minitorch.Adam(self.model.parameters(), learning_rate)
        
        losses = []
        train_accuracy = []
        validation_accuracy = []

        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            n_batches = 0
            
            model.train()  # enable training mode
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            
            # -----------------------------------------------------------------
            # Training Loop Over Batches
            # -----------------------------------------------------------------
            for start_idx in range(0, n_training_samples, batch_size):
                end_idx = start_idx + batch_size
                x_data = X_train[start_idx:end_idx]
                y_data = y_train[start_idx:end_idx]

                # BEGIN ASSIGN1_4
                # 1) Create x and y as minitorch tensors, with requires_grad=True
                x = minitorch.tensor(x_data, backend=BACKEND, requires_grad=True)
                y = minitorch.tensor(y_data, backend=BACKEND, requires_grad=True)

                # 2) Forward pass -> out
                out = model(x)

                # 3) Calculate loss
                # loss = minitorch.binary_cross_entropy(out, y)
                loss = binary_cross_entropy(out, y)  # <--- custom BCE

                # 4) Zero out old gradients
                optim.zero_grad()

                # 5) Backprop
                loss.backward()

                # 6) Use optimizer to take a step
                optim.step()
                # END ASSIGN1_4

                # Save training results
                train_predictions += get_predictions_array(y, out)
                total_loss += float(loss[0])
                n_batches += 1

            # -----------------------------------------------------------------
            # Validation Step
            # -----------------------------------------------------------------
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()  # disable dropout, etc.

                # BEGIN ASSIGN1_4
                # 1) Create x and y as minitorch tensors (no grad needed)
                x_val = minitorch.tensor(X_val, backend=BACKEND)
                y_val_t = minitorch.tensor(y_val, backend=BACKEND)

                # 2) Forward pass -> out_val
                out_val = model(x_val)

                # 3) Obtain validation predictions
                val_predictions = get_predictions_array(y_val_t, out_val)
                validation_predictions += val_predictions

                # 4) Obtain the validation accuracy
                val_acc = get_accuracy(val_predictions)
                validation_accuracy.append(val_acc)
                # END ASSIGN1_4

                model.train()  # switch back to training mode

            # Compute training accuracy this epoch
            epoch_train_acc = get_accuracy(train_predictions)
            train_accuracy.append(epoch_train_acc)

            # Average loss across all batches
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            losses.append(avg_loss)

            # Log everything
            log_fn(
                epoch,
                avg_loss,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )

def binary_cross_entropy(pred, target, eps=1e-6):
    """
    pred, target: Tensors of shape (batch,)
    Return a scalar Tensor representing BCE:
        BCE = -( y * log(pred+eps) + (1-y) * log(1-pred+eps) ).mean()
    """
    # Create a 'one' Tensor with the same shape as pred or target
    one = pred.zeros() + 1.0  # shape (batch,)

    # term1 = y * log(pred + eps)
    term1 = target * (pred + eps).log()

    # term2 = (1 - y) * log((1 - pred) + eps)
    term2 = (one - target) * ((one - pred) + eps).log()

    # sum them, average, negate
    out_mean = (term1 + term2).mean()
    return out_mean * -1.0


# ===================================================================
# Data Encoding & Execution
# ===================================================================
def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0s to max sentence length in order to enable batching
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                # use random embedding for unks
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    # load labels
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    #  Determine max sentence length
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for _ in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    from embeddings import GloveEmbedding

    train_size = 450 #450
    validation_size = 100 #100
    learning_rate = 0.25 #0.25
    max_epochs = 250 #250
    embedding_dim = 50 #50

    dataset = load_dataset("glue", "sst2")
    pretrained_embeddings = GloveEmbedding("wikipedia_gigaword", d_emb=embedding_dim, show_progress=True)

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        dataset,
        pretrained_embeddings,
        train_size,
        validation_size,
    )

    model_trainer = SentenceSentimentTrain()
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        batch_size=BATCH,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )
    

def generate(model,
             examples,
             src_key,
             tgt_key,
             tokenizer,
             model_max_length,
             backend,
             desc):
    """
    Generates target sequences (English) from source sequences (German),
    by repeatedly argmax-decoding the next token.

    For each example:
      1) Tokenize the source text + <eos_de>.
      2) Repeatedly run the model to get next-token logits, take argmax,
         and append it to token_ids.
      3) Stop if we see <eos_en> or exceed model_max_length.
      4) Decode tokens from the end of the source onward.

    Returns a list of generated target sentences (strings).
    """
    import tqdm
    model.eval()
    gen_sents = []

    # Disable gradient tracking in generation mode
    with minitorch.no_grad():
        for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
            # 1) Tokenize the German source + <eos_de>
            token_ids = tokenizer(f"{example[src_key]}<eos_{src_key}>")['input_ids']
            len_src = len(token_ids)

            # 2) Repeatedly predict next token until <eos_en> or max length
            while len(token_ids) < model_max_length:
                # shape: [1, current_length]
                idx = minitorch.tensor(token_ids, backend=backend).view(1, -1)
                logits = model(idx=idx)  # => shape [1, seq_len, vocab_size]

                # Take the last position's logits => shape [vocab_size]
                last_logits = logits[0, -1, :]

                # Argmax to pick the highest-prob token
                next_id = last_logits.argmax()
                gen_id = int(next_id.item())

                # Stop if it's the end-of-sentence token for English
                if gen_id == tokenizer.vocab[f"<eos_{tgt_key}>"]:
                    break

                token_ids.append(gen_id)

            # 3) Decode from just after source tokens => predicted target
            gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents



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

# import minitorch
# from minitorch import DecoderLM
# from minitorch.cuda_kernel_ops import CudaKernelOps


# def get_dataset(dataset_name, model_max_length):
#     """
#     Obtrain IWSLT (de-en) dataset.
#     """
#     dataset = {
#         split: datasets.load_dataset(dataset_name, split=split)['translation']
#         for split in ['train', 'validation', 'test']
#     }
#     src_key, tgt_key = 'de', 'en'

#     dataset = {
#         split: [
#             example for example in dataset[split]
#             if len(example[src_key].split()) + len(
#                 example[tgt_key].split()) < model_max_length
#         ] for split in dataset.keys()
#     }

#     dataset['test'] = dataset['test'][:100]  # 6750

#     print(json.dumps(
#         {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
#         indent=4))

#     return dataset, src_key, tgt_key


# def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
#     """
#     Trains a tokenizer on the provided dataset examples and saves the tokenizer configuration.

#     Parameters:
#     - examples: The dataset examples used for training the tokenizer.
#     - vocab_size: The desired vocabulary size for the tokenizer.
#     - src_key: The key used to access the source text within the dataset examples.
#     - tgt_key: The key used to access the target text within the dataset examples.
#     - workdir: The directory where the tokenizer should be saved.

#     Returns:
#     - tokenizer: The trained tokenizer with special tokens,
#         e.g., ("<eos_de>", "<eos_en>", "<pad>") if src_key and tgt_key are "de" and "en", respectively.
#     """
#     tokenizer = ByteLevelBPETokenizer()

#     # Customized training
#     tokenizer.train_from_iterator(
#         [[example[src_key], example[tgt_key]] for example in examples],
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

#     Parameters:
#     - examples: A list of examples to be processed.
#     - src_key: The key for accessing source texts in the examples.
#     - tgt_key: The key for accessing target texts in the examples.
#     - tokenizer: The tokenizer to be used for encoding the texts.
#     - model_max_length: The maximum sequence length the model can handle.
#     - backend: The backend of minitorch tensors.

#     Returns:
#     - A dictionary containing keys: 'input_ids', 'labels', 'label_token_weights',
#         each indicates a minitorch tensor with shape (len(examples), model_max_length).

#     Notes:
#     ["input_ids"] for every example in the DE-EN translation, the "input_ids" will be:
#         <de_token_ids> + <de_eos_id> + <en_token_ids> + <en_eos_id> + <pad_ids>
#     where the pad_ids makes the length of input_ids to be model_max_length.

#     ["labels"]: the next tokens to be predicted, which will be used in the cross-entropy
#     loss function, e.g., for an example tokenized as [a, b, c, d], "input_ids" and "labels" 
#     can be [a, b, c] and [b, c, d], respectively.

#     ["label_token_weights"] The 'label_token_weights' are used to differentiate
#     calculation purposes. (the MLE loss is computed on target tokens only.)
#     between the source (weight = 0) and target (weight = 1) tokens for loss
#     """
#     token_ids, tgt_token_mask = [], []
#     max_length = model_max_length
#     pad_token_id = tokenizer.vocab['<pad>']
#     for example in examples:
#         token_ids_src = tokenizer(
#             f'{example[src_key]}<eos_{src_key}>')['input_ids']
#         token_ids_tgt = tokenizer(
#             f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

#         example_token_ids = token_ids_src + token_ids_tgt
#         example_tgt_token_mask = (
#                 [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
#         example_token_ids = example_token_ids[:max_length]
#         example_tgt_token_mask = example_tgt_token_mask[:max_length]
#         pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

#         token_ids.append(example_token_ids + pad_ids)
#         tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

#     # TODO: make examples in a 1d list, provide shape to initialize minitorch.Tensor
#     token_ids = np.array(token_ids)
#     tgt_token_mask = np.array(tgt_token_mask)

#     input_ids = token_ids[:, :-1]
#     labels    = token_ids[:, 1:]
#     label_token_weights = tgt_token_mask[:, 1:]

#     input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
#     labels    = minitorch.tensor_from_numpy(labels, backend=backend)
#     label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)
    
#     # input_ids = token_ids[:, :-1].tolist()
#     # labels    = token_ids[:, 1:].tolist()
#     # label_token_weights = tgt_token_mask[:, 1:].tolist()

#     # input_ids = minitorch.tensor(input_ids, backend=backend)
#     # labels    = minitorch.tensor(labels, backend=backend)
#     # label_token_weights = minitorch.tensor(label_token_weights, backend=backend)

#     return {
#         'input_ids': input_ids,
#         'labels': labels,
#         'label_token_weights': label_token_weights
#     }


# def loss_fn(batch, model):
#     """
#     The MLE loss for a batch.

#     Parameters:
#     - batch: The result of collate_fn, a dict with "input_ids", "labels", and "label_token_weights".
#     - model: The model to be trained.

#     Returns:
#     - A scalar loss value for this batch, averaged across all target tokens.
#     """

#     idx = batch['input_ids']
#     idx.requires_grad_(True)
#     # print("getting into loss_fn")
#     logits = model(idx=idx)
#     # print("finish prediction")
#     bs, l, c = logits.shape
#     logits = logits.view(bs * l, c)
#     targets = batch['labels'].view(bs * l)
#     label_token_weights = batch['label_token_weights'].view(bs * l)

#     targets.requires_grad_(True)
#     # print("start calculating loss")
#     # import pdb
#     # pdb.set_trace()
#     loss = minitorch.nn.softmax_loss(
#         logits=logits,
#         target=targets
#     )

#     return ((loss * label_token_weights).sum() / label_token_weights.sum())


# def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
#     """
#     Trains the model on the provided examples.

#     Parameters:
#     - model: The model to be trained.
#     - optimizer: The optimizer used for updating the model's parameters.
#     - examples: The dataset examples used for training.
#     - n_samples: The random samples to train from "examples".
#     - collate_fn: The function to collate data examples into batches.
#     - batch_size: The number of examples in each batch.
#     - desc: Description for the training process (used in progress bars).
#     """
#     model.train()
#     random.shuffle(examples)
#     examples = examples[:n_samples]

#     for i in (prog_bar := tqdm.trange(
#             0, len(examples), batch_size, desc=f'Training ({desc})')):
#         batch = collate_fn(examples=examples[i:i + batch_size])

#         t0 = time.time()
#         optimizer.zero_grad()
#         loss = loss_fn(batch=batch, model=model)
#         t1 = time.time()

#         loss.backward()
#         t2 = time.time()

#         optimizer.step()
#         t3 = time.time()

#         print(f"Forward: {t1 - t0}")
#         print(f"Backward: {t2 - t1}")
#         print(f"Opt.step: {t3 - t2}")

#         batch_time = time.time() - t0
#         prog_bar.set_postfix(
#             tokens_per_sec=np.prod(batch['input_ids'].shape) / batch_time,
#             loss=loss.item(),
#             lr=optimizer.lr)


# def evaluate_loss(model, examples, batch_size, collate_fn, desc):
#     """
#     Evaluates the model on the provided examples and computes the average loss.

#     Parameters:
#     - model: The model to be evaluated.
#     - examples: The dataset examples used for evaluation.
#     - batch_size: The number of examples in each batch.
#     - collate_fn: The function to collate data examples into batches.
#     - desc: Description for the evaluation process (used in progress bars).

#     Returns:
#     - The average loss computed over all batches.
#     """
#     model.eval()
#     losses = []

#     for i in (prog_bar := tqdm.trange(
#         0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
#         batch = collate_fn(examples=examples[i:i + batch_size])
#         loss = loss_fn(batch=batch, model=model)

#         losses.append(loss.item())
#         prog_bar.set_postfix(loss=loss.item())

#     return np.mean(losses)


# def generate(model,
#              examples,
#              src_key,
#              tgt_key,
#              tokenizer,
#              model_max_length,
#              backend,
#              desc):
#     """
#     Generates target sequences for the given source sequences using the model, based on argmax decoding.
#     Note that it runs generation on examples one-by-one instead of in a batched manner.

#     Parameters:
#     - model: The model used for generation.
#     - examples: The dataset examples containing source sequences.
#     - src_key: The key for accessing source texts in the examples.
#     - tgt_key: The key for accessing target texts in the examples.
#     - tokenizer: The tokenizer used for encoding texts.
#     - model_max_length: The maximum sequence length the model can handle.
#     - backend: The backend of minitorch tensors.
#     - desc: Description for the generation process (used in progress bars).

#     Returns:
#     - A list of generated target sequences.
#     """

#     model.eval()
#     gen_sents = []
#     for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
#         # Run generation for every single example

#         token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
#         len_src = len(token_ids)

#         while len(token_ids) <= model_max_length:
#             # BEGIN ASSIGN2_2
#             # TODO
#             # run the model with current token_ids, and predict the next token (gen_id)
#             # hint: obtain the logits of next token, and take the argmax.
#             gen_id = 0
#             raise NotImplementedError("Generation Function Not Implemented Yet")
#             # END ASSIGN2_2

#             if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
#                 break
#             else:
#                 token_ids.append(gen_id)

#         gen_sents.append(tokenizer.decode(token_ids[len_src:]))

#     return gen_sents


# def evaluate_bleu(examples, gen_sents, tgt_key):
#     """
#     Evaluates the BLEU score for generated sentences against the target sentences in the examples.

#     Parameters:
#     - examples: The dataset examples used for evaluation.
#     - gen_sents: The generated sentences to be evaluated.
#     - tgt_key: The key for accessing target texts in the examples.

#     Returns:
#     - A dictionary containing the BLEU score.
#     """
#     return {
#         'bleu': BLEU().corpus_score(
#             hypotheses=gen_sents,
#             references=[[example[tgt_key] for example in examples]]).score
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

#     Parameters:
#     - dataset_name: The name of the dataset to be used.
#     - model_max_length: The maximum sequence length the model can handle.
#     - n_epochs: The number of training epochs.
#     - batch_size: The number of examples in each batch.
#     - learning_rate: The learning rate for the optimizer.
#     - samples_per_epoch: Samples from the training dataset every epoch.
#     - n_vocab: The vocabulary size of the BPE tokenizer.
#     - n_embd: The embedding dimension.
#     - seed: Random seed.
#     """

#     np.random.seed(seed)
#     random.seed(seed)

#     workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
#     os.makedirs(workdir, exist_ok=True)

#     backend = minitorch.TensorBackend(CudaKernelOps)

#     config = {
#         'n_vocab': n_vocab,  # vocab_size
#         'n_embd': n_embd,  # n_embed
#         'n_head': 8,  # n_head
#         'n_positions': model_max_length,  # n_ctx == n_positions
#         # 'n_layer'     : 4,    # n_layer
#         'p_dropout': 0.1,  # x_pdrop
#         'ln_eps': 1e-5,  # layer_norm_epsilon
#         'backend': backend
#     }

#     model = DecoderLM(**config)
#     optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

#     dataset, src_key, tgt_key = get_dataset(
#         dataset_name=dataset_name, model_max_length=model_max_length)

#     tokenizer = get_tokenizer(
#         examples=dataset['train'],
#         vocab_size=config['n_vocab'],
#         src_key=src_key,
#         tgt_key=tgt_key,
#         workdir=workdir)

#     collate_fn = partial(
#         collate_batch,
#         src_key=src_key,
#         tgt_key=tgt_key,
#         tokenizer=tokenizer,
#         model_max_length=model_max_length,
#         backend=backend)

#     for epoch_idx in range(n_epochs):
#         desc = f'epoch {epoch_idx} / {n_epochs}'

#         train(
#             model=model,
#             optimizer=optimizer,
#             examples=dataset['train'],
#             n_samples=samples_per_epoch,
#             batch_size=batch_size,
#             collate_fn=collate_fn,
#             desc=desc)

#         validation_loss = evaluate_loss(
#             model=model,
#             examples=dataset['validation'],
#             batch_size=batch_size,
#             collate_fn=collate_fn,
#             desc=desc)

#         print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss}')

#         gen_sents = generate(
#             model=model,
#             examples=dataset['test'],
#             src_key=src_key,
#             tgt_key=tgt_key,
#             tokenizer=tokenizer,
#             model_max_length=model_max_length,
#             backend=backend,
#             desc=desc)

#         gen_examples = []
#         for example, gen_sent in zip(dataset['test'], gen_sents):
#             gen_examples.append({'example': example, 'gen': gen_sent})
#         json.dump(gen_examples, open(
#             f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

#         eval_scores = evaluate_bleu(
#             examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
#         print(f'Epoch {epoch_idx}: {eval_scores}')

#         json.dump(
#             {'validation_loss': float(validation_loss), **eval_scores},
#             open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))


# if __name__ == '__main__':
#     fire.Fire(main)
