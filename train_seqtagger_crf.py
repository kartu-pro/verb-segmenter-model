"""
train_seqtagger.py
------------------
PyTorch training pipeline for Sequence Tagging models with CRF layer.

Integrates:
  - Character Embeddings + Metadata Embeddings (screeve, subject, verb_class).
  - BiLSTM sequence encoder + Linear-Chain CRF output layer.
  - Supports 'boundaries' and 'bio' strategies.
"""

import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

import load
import seq_tagger
import evaluate
import experiments


# ==========================================
# 1. PYTORCH DATASET & COLLATOR
# ==========================================

class GeorgianTaggingDataset(Dataset):
    """PyTorch Dataset wrapping encoded integer IDs."""

    def __init__(self, encoded_df: pd.DataFrame):
        # Filter out empty sequences to prevent zero-length RNN errors
        if "token_ids" in encoded_df.columns:
            encoded_df = encoded_df[encoded_df["token_ids"].apply(
                lambda x: len(x) > 0)].reset_index(drop=True)
        elif "seq_len" in encoded_df.columns:
            encoded_df = encoded_df[encoded_df["seq_len"]
                                    > 0].reset_index(drop=True)

        self.data = encoded_df.to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "token_ids": torch.tensor(item["token_ids"], dtype=torch.long),
            "label_ids": torch.tensor(item["label_ids"], dtype=torch.long),
            "screeve_id": torch.tensor(item["screeve_id"], dtype=torch.long),
            "subject_id": torch.tensor(item["subject_id"], dtype=torch.long),
            "class_id": torch.tensor(item["class_id"], dtype=torch.long),
            "seq_len": item["seq_len"]
        }


def pad_collate_fn(batch):
    """Pads character sequences and labels dynamically per batch."""
    token_ids = [b["token_ids"] for b in batch]
    label_ids = [b["label_ids"] for b in batch]

    padded_tokens = nn.utils.rnn.pad_sequence(
        token_ids, batch_first=True, padding_value=0)
    padded_labels = nn.utils.rnn.pad_sequence(
        label_ids, batch_first=True, padding_value=0)  # Use 0 or appropriate padding for CRF labels

    screeve_ids = torch.stack([b["screeve_id"] for b in batch])
    subject_ids = torch.stack([b["subject_id"] for b in batch])
    class_ids = torch.stack([b["class_id"] for b in batch])

    return {
        "tokens": padded_tokens,
        "labels": padded_labels,
        "screeve_id": screeve_ids,
        "subject_id": subject_ids,
        "class_id": class_ids
    }


# ==========================================
# 2. BiLSTM METADATA TAGGER MODEL WITH CRF
# ==========================================

class BiLSTMMetadataTaggerWithCRF(nn.Module):
    """BiLSTM Model with metadata feature concatenation and CRF output layer."""

    def __init__(self, vocabs: dict, char_dim: int = 64, meta_dim: int = 16, hidden_dim: int = 128):
        super().__init__()

        self.char_embed = nn.Embedding(
            len(vocabs["char2id"]), char_dim, padding_idx=0)
        self.screeve_embed = nn.Embedding(len(vocabs["screeve2id"]), meta_dim)
        self.subject_embed = nn.Embedding(len(vocabs["subject2id"]), meta_dim)
        self.class_embed = nn.Embedding(len(vocabs["class2id"]), meta_dim)

        input_dim = char_dim + (meta_dim * 3)
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, len(vocabs["label2id"]))

        # Linear-chain CRF layer
        self.crf = CRF(len(vocabs["label2id"]), batch_first=True)

    def forward(self, tokens, screeve_id, subject_id, class_id, labels=None, mask=None):
        batch_size, seq_len = tokens.shape

        # Lookup embeddings
        c_emb = self.char_embed(tokens)
        s_emb = self.screeve_embed(screeve_id).unsqueeze(
            1).expand(-1, seq_len, -1)
        subj_emb = self.subject_embed(
            subject_id).unsqueeze(1).expand(-1, seq_len, -1)
        cls_emb = self.class_embed(class_id).unsqueeze(
            1).expand(-1, seq_len, -1)

        # Concatenate features
        combined = torch.cat([c_emb, s_emb, subj_emb, cls_emb], dim=-1)

        # Encode & Emission scores
        lstm_out, _ = self.lstm(combined)
        emissions = self.fc(lstm_out)

        if mask is None:
            mask = (tokens != 0)

        if labels is not None:
            # Returns negative log-likelihood loss for training
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            # Returns Viterbi decoded best path during evaluation/inference
            return self.crf.decode(emissions, mask=mask)


# ==========================================
# 3. TRAINING LOOP
# ==========================================

def run_training(
    words_path: str = "words.tsv",
    forms_path: str = "forms.tsv",
    strategy: str = "boundaries",
    epochs: int = 10,
    batch_size: int = 16,
    max_samples: int = None,
):
    start_time = time.time()
    print(
        f"--- Starting Sequence Tagger Training with CRF (Strategy: {strategy}) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data & Build Vocabularies
    raw_df = load.load_and_merge_data(words_path, forms_path)

    if max_samples:
        raw_df = raw_df.head(max_samples)
        print(
            f"⚠️ [Fast Mode] Truncated dataset to {max_samples} samples for quick verification.")

    # Split by verb
    unique_words = raw_df["word_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_words)
    n_train = int(len(unique_words) * 0.8)
    n_val = int(len(unique_words) * 0.1)

    train_df = raw_df[raw_df["word_id"].isin(set(unique_words[:n_train]))]
    val_df = raw_df[raw_df["word_id"].isin(
        set(unique_words[n_train:n_train + n_val]))]
    test_df = raw_df[raw_df["word_id"].isin(
        set(unique_words[n_train + n_val:]))]

    vocabs = seq_tagger.build_vocabularies(raw_df, strategy=strategy)

    # 2. Dataset & Loaders
    train_encoded = seq_tagger.encode_with_ids(
        train_df, vocabs, strategy=strategy)
    val_encoded = seq_tagger.encode_with_ids(val_df, vocabs, strategy=strategy)
    test_encoded = seq_tagger.encode_with_ids(
        test_df, vocabs, strategy=strategy)

    train_loader = DataLoader(GeorgianTaggingDataset(
        train_encoded), batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(GeorgianTaggingDataset(
        val_encoded), batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(GeorgianTaggingDataset(
        test_encoded), batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    # 3. Model & Optimizer
    model = BiLSTMMetadataTaggerWithCRF(vocabs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Reverse label lookup
    id2label = {v: k for k, v in vocabs["label2id"].items()}
    id2char = {v: k for k, v in vocabs["char2id"].items()}

    # 4. Epoch Loop with Validation F1 Tracking
    history = {"epoch": [], "train_loss": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)
            screeve_id = batch["screeve_id"].to(device)
            subject_id = batch["subject_id"].to(device)
            class_id = batch["class_id"].to(device)
            mask = (tokens != 0)

            loss = model(tokens, screeve_id, subject_id,
                         class_id, labels=labels, mask=mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)

        # --- Validation Evaluation per Epoch ---
        model.eval()
        val_pred_strings = []
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                screeve_id = batch["screeve_id"].to(device)
                subject_id = batch["subject_id"].to(device)
                class_id = batch["class_id"].to(device)
                mask = (tokens != 0)

                predicted_paths = model(
                    tokens, screeve_id, subject_id, class_id, labels=None, mask=mask)
                tok_ids = tokens.cpu().numpy()

                for b_idx, preds in enumerate(predicted_paths):
                    valid_mask = tok_ids[b_idx] != 0
                    valid_toks = [id2char[t]
                                  for t in tok_ids[b_idx][valid_mask]]
                    valid_preds = [id2label[p] for p in preds]
                    val_pred_strings.append(seq_tagger.decode(
                        valid_toks, valid_preds, strategy=strategy))

        val_gold = val_df["segmented_target"].tolist()
        val_metrics = evaluate.evaluate_predictions(val_gold, val_pred_strings)
        val_f1 = val_metrics.get("morpheme_f1", 0.0)

        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} | Val Morpheme F1: {val_f1:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocabs": vocabs,
                "strategy": strategy,
                "epoch": epoch,
                "val_f1": best_f1,
            }
            torch.save(checkpoint, f"model_tagger_crf_{strategy}.pt")
            print(
                f"💾 [Checkpoint] Saved new best CRF model with Val Morpheme F1: {best_f1:.4f} at epoch {epoch}")

    # 5. Final Test Set Evaluation (Using the Best Saved Checkpoint)
    print("\n--- Loading Best Checkpoint for Test Set Evaluation ---")
    checkpoint = torch.load(
        f"model_tagger_crf_{strategy}.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_pred_strings = []
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch["tokens"].to(device)
            screeve_id = batch["screeve_id"].to(device)
            subject_id = batch["subject_id"].to(device)
            class_id = batch["class_id"].to(device)
            mask = (tokens != 0)

            predicted_paths = model(
                tokens, screeve_id, subject_id, class_id, labels=None, mask=mask)
            tok_ids = tokens.cpu().numpy()

            for b_idx, preds in enumerate(predicted_paths):
                valid_mask = tok_ids[b_idx] != 0
                valid_toks = [id2char[t] for t in tok_ids[b_idx][valid_mask]]
                valid_preds = [id2label[p] for p in preds]
                test_pred_strings.append(seq_tagger.decode(
                    valid_toks, valid_preds, strategy=strategy))

    gold_strings = test_df["segmented_target"].tolist()
    metrics = evaluate.evaluate_predictions(gold_strings, test_pred_strings)
    elapsed_time = time.time() - start_time
    metrics["training_time_sec"] = round(elapsed_time, 1)

    input_texts = test_df["form"].tolist()
    evaluate.export_error_analysis(
        input_texts,
        gold_strings,
        test_pred_strings,
        output_path=f"error_analysis_tagger_crf_{strategy}_crf.csv",
    )

    with open(f"./results_tagger_crf_{strategy}_crf_history.json", "w") as f:
        json.dump(history, f, indent=4)
    experiments.log_experiment("BiLSTM-Meta-CRF", strategy, metrics)

    print("\n--- Test Set Evaluation ---")
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words_path", type=str, default="words.tsv")
    parser.add_argument("--forms_path", type=str, default="forms.tsv")
    parser.add_argument("--strategy", type=str,
                        choices=["boundaries", "bio"], default="bio")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_training(args.words_path, args.forms_path,
                 strategy=args.strategy, epochs=args.epochs)
