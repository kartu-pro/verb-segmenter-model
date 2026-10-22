"""
train_seqtagger.py
------------------
Unified PyTorch training pipeline for Sequence Tagging models (Standard BiLSTM & BiLSTM-CRF).

Integrates:
  - Character Embeddings + Metadata Embeddings (screeve, subject, verb_class).
  - BiLSTM sequence encoder + Optional Linear-Chain CRF output layer.
  - Supports 'boundaries' and 'bio' strategies via command line flags.
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
from utils import split_data_by_verb


# ==========================================
# 1. PYTORCH DATASET & COLLATOR
# ==========================================

class GeorgianTaggingDataset(Dataset):
    """PyTorch Dataset wrapping encoded integer IDs."""

    def __init__(self, encoded_df: pd.DataFrame):
        if "token_ids" in encoded_df.columns:
            encoded_df = encoded_df[encoded_df["token_ids"].apply(
                lambda x: len(x) > 0)].reset_index(drop=True)
        elif "seq_len" in encoded_df.columns:
            encoded_df = encoded_df[encoded_df["seq_len"] > 0].reset_index(drop=True)

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


def get_collate_fn(use_crf: bool):
    """Returns a collate function configured for standard CrossEntropy or CRF padding labels."""
    label_pad_value = 0 if use_crf else -100

    def pad_collate_fn(batch):
        token_ids = [b["token_ids"] for b in batch]
        label_ids = [b["label_ids"] for b in batch]

        padded_tokens = nn.utils.rnn.pad_sequence(
            token_ids, batch_first=True, padding_value=0)
        padded_labels = nn.utils.rnn.pad_sequence(
            label_ids, batch_first=True, padding_value=label_pad_value)

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
    return pad_collate_fn


# ==========================================
# 2. MODEL DEFINITIONS (Standard & CRF)
# ==========================================

class BiLSTMMetadataTagger(nn.Module):
    """BiLSTM Model with metadata feature concatenation."""

    def __init__(self, vocabs: dict, char_dim: int = 64, meta_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.char_embed = nn.Embedding(len(vocabs["char2id"]), char_dim, padding_idx=0)
        self.screeve_embed = nn.Embedding(len(vocabs["screeve2id"]), meta_dim)
        self.subject_embed = nn.Embedding(len(vocabs["subject2id"]), meta_dim)
        self.class_embed = nn.Embedding(len(vocabs["class2id"]), meta_dim)

        input_dim = char_dim + (meta_dim * 3)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, len(vocabs["label2id"]))

    def forward(self, tokens, screeve_id, subject_id, class_id):
        batch_size, seq_len = tokens.shape
        c_emb = self.char_embed(tokens)
        s_emb = self.screeve_embed(screeve_id).unsqueeze(1).expand(-1, seq_len, -1)
        subj_emb = self.subject_embed(subject_id).unsqueeze(1).expand(-1, seq_len, -1)
        cls_emb = self.class_embed(class_id).unsqueeze(1).expand(-1, seq_len, -1)

        combined = torch.cat([c_emb, s_emb, subj_emb, cls_emb], dim=-1)
        lstm_out, _ = self.lstm(combined)
        logits = self.fc(lstm_out)
        return logits


class BiLSTMMetadataTaggerWithCRF(nn.Module):
    """BiLSTM Model with metadata feature concatenation and CRF output layer."""

    def __init__(self, vocabs: dict, char_dim: int = 64, meta_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.char_embed = nn.Embedding(len(vocabs["char2id"]), char_dim, padding_idx=0)
        self.screeve_embed = nn.Embedding(len(vocabs["screeve2id"]), meta_dim)
        self.subject_embed = nn.Embedding(len(vocabs["subject2id"]), meta_dim)
        self.class_embed = nn.Embedding(len(vocabs["class2id"]), meta_dim)

        input_dim = char_dim + (meta_dim * 3)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, len(vocabs["label2id"]))
        self.crf = CRF(len(vocabs["label2id"]), batch_first=True)

    def forward(self, tokens, screeve_id, subject_id, class_id, labels=None, mask=None):
        batch_size, seq_len = tokens.shape
        c_emb = self.char_embed(tokens)
        s_emb = self.screeve_embed(screeve_id).unsqueeze(1).expand(-1, seq_len, -1)
        subj_emb = self.subject_embed(subject_id).unsqueeze(1).expand(-1, seq_len, -1)
        cls_emb = self.class_embed(class_id).unsqueeze(1).expand(-1, seq_len, -1)

        combined = torch.cat([c_emb, s_emb, subj_emb, cls_emb], dim=-1)
        lstm_out, _ = self.lstm(combined)
        emissions = self.fc(lstm_out)

        if mask is None:
            mask = (tokens != 0)

        if labels is not None:
            return -self.crf(emissions, labels, mask=mask, reduction='mean')
        else:
            return self.crf.decode(emissions, mask=mask)


# ==========================================
# 3. TRAINING LOOP
# ==========================================

def run_training(
    words_path: str = "words.tsv",
    forms_path: str = "forms.tsv",
    strategy: str = "boundaries",
    use_crf: bool = False,
    epochs: int = 10,
    batch_size: int = 16,
    max_samples: int = None,
):
    model_type_label = "BiLSTM-Meta-CRF" if use_crf else "BiLSTM-Meta"
    print(f"--- Starting Sequence Tagger Training ({model_type_label}, Strategy: {strategy}) ---")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data & Split
    raw_df = load.load_and_merge_data(words_path, forms_path)
    if max_samples:
        raw_df = raw_df.head(max_samples)
        print(f"⚠️ [Fast Mode] Truncated dataset to {max_samples} samples.")

    train_df, val_df, test_df = split_data_by_verb(raw_df)
    vocabs = seq_tagger.build_vocabularies(raw_df, strategy=strategy)

    # 2. Dataset & Loaders
    train_encoded = seq_tagger.encode_with_ids(train_df, vocabs, strategy=strategy)
    val_encoded = seq_tagger.encode_with_ids(val_df, vocabs, strategy=strategy)
    test_encoded = seq_tagger.encode_with_ids(test_df, vocabs, strategy=strategy)

    collate_fn = get_collate_fn(use_crf)
    train_loader = DataLoader(GeorgianTaggingDataset(train_encoded), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(GeorgianTaggingDataset(val_encoded), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(GeorgianTaggingDataset(test_encoded), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Model & Optimizer Setup
    if use_crf:
        model = BiLSTMMetadataTaggerWithCRF(vocabs).to(device)
        checkpoint_filename = f"model_tagger_crf_{strategy}.pt"
    else:
        model = BiLSTMMetadataTagger(vocabs).to(device)
        checkpoint_filename = f"model_tagger_{strategy}.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = None if use_crf else nn.CrossEntropyLoss(ignore_index=-100)

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

            if use_crf:
                mask = (tokens != 0)
                loss = model(tokens, screeve_id, subject_id, class_id, labels=labels, mask=mask)
            else:
                logits = model(tokens, screeve_id, subject_id, class_id)
                loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

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

                if use_crf:
                    predicted_paths = model(tokens, screeve_id, subject_id, class_id, labels=None, mask=mask)
                    tok_ids = tokens.cpu().numpy()
                    for b_idx, preds in enumerate(predicted_paths):
                        valid_mask = tok_ids[b_idx] != 0
                        valid_toks = [id2char[t] for t in tok_ids[b_idx][valid_mask]]
                        valid_preds = [id2label[p] for p in preds]
                        val_pred_strings.append(seq_tagger.decode(valid_toks, valid_preds, strategy=strategy))
                else:
                    logits = model(tokens, screeve_id, subject_id, class_id)
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    tok_ids = tokens.cpu().numpy()
                    for b_idx in range(len(preds)):
                        valid_mask = tok_ids[b_idx] != 0
                        valid_toks = [id2char[t] for t in tok_ids[b_idx][valid_mask]]
                        valid_preds = [id2label[p] for p in preds[b_idx][valid_mask]]
                        val_pred_strings.append(seq_tagger.decode(valid_toks, valid_preds, strategy=strategy))

        val_gold = val_df["segmented_target"].tolist()
        val_metrics = evaluate.evaluate_predictions(val_gold, val_pred_strings)
        val_f1 = val_metrics.get("morpheme_f1", 0.0)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} | Val Morpheme F1: {val_f1:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocabs": vocabs,
                "strategy": strategy,
                "use_crf": use_crf,
                "epoch": epoch,
                "val_f1": best_f1,
            }
            torch.save(checkpoint, checkpoint_filename)
            print(f"💾 [Checkpoint] Saved new best model with Val Morpheme F1: {best_f1:.4f} at epoch {epoch}")

    # 5. Final Test Set Evaluation (Using Best Checkpoint)
    print("\n--- Loading Best Checkpoint for Test Set Evaluation ---")
    checkpoint = torch.load(checkpoint_filename, map_location=device)
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

            if use_crf:
                predicted_paths = model(tokens, screeve_id, subject_id, class_id, labels=None, mask=mask)
                tok_ids = tokens.cpu().numpy()
                for b_idx, preds in enumerate(predicted_paths):
                    valid_mask = tok_ids[b_idx] != 0
                    valid_toks = [id2char[t] for t in tok_ids[b_idx][valid_mask]]
                    valid_preds = [id2label[p] for p in preds]
                    test_pred_strings.append(seq_tagger.decode(valid_toks, valid_preds, strategy=strategy))
            else:
                logits = model(tokens, screeve_id, subject_id, class_id)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                tok_ids = tokens.cpu().numpy()
                for b_idx in range(len(preds)):
                    valid_mask = tok_ids[b_idx] != 0
                    valid_toks = [id2char[t] for t in tok_ids[b_idx][valid_mask]]
                    valid_preds = [id2label[p] for p in preds[b_idx][valid_mask]]
                    test_pred_strings.append(seq_tagger.decode(valid_toks, valid_preds, strategy=strategy))

    gold_strings = test_df["segmented_target"].tolist()
    metrics = evaluate.evaluate_predictions(gold_strings, test_pred_strings)
    elapsed_time = time.time() - start_time
    metrics["training_time_sec"] = round(elapsed_time, 1)

    experiments.log_experiment(
        model_name=model_type_label,
        strategy_mode=strategy,
        metrics=metrics,
        history=history,
        error_inputs=test_df["form"].tolist(),
        error_golds=gold_strings,
        error_preds=test_pred_strings,
        error_output_path=f"error_analysis_tagger_crf_{strategy}.csv" if use_crf else f"error_analysis_tagger_{strategy}.csv"
    )

    print("\n--- Test Set Evaluation ---")
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words_path", type=str, default="words.tsv")
    parser.add_argument("--forms_path", type=str, default="forms.tsv")
    parser.add_argument("--strategy", type=str, choices=["boundaries", "bio"], default="boundaries")
    parser.add_argument("--use_crf", action="store_true", help="Use Linear-Chain CRF output layer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_training(
        words_path=args.words_path,
        forms_path=args.forms_path,
        strategy=args.strategy,
        use_crf=args.use_crf,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
