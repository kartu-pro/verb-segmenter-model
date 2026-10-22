"""
train_seqtagger.py
------------------
Unified PyTorch training pipeline for Sequence Tagging models (Standard BiLSTM & BiLSTM-CRF).
"""

import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import load
import seq_tagger
import evaluate
import experiments
from utils import split_data_by_verb
from tagger_models import BiLSTMMetadataTagger, BiLSTMMetadataTaggerWithCRF
from tagger_datasets import GeorgianTaggingDataset, get_collate_fn


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