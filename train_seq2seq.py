"""
train_seq2seq.py
----------------
Training pipeline for Sequence-to-Sequence models using HuggingFace Transformers & ByT5/T5.

Supports both 'single' word and 'group6' paradigm prompt modes.
Runs verb-grouped train/val/test splits to avoid data leakage.
"""

import time
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

import load
import seq_2_seq
import evaluate
import experiments

import os
import shutil
import glob
from transformers import TrainerCallback


def restore_latest_checkpoint_from_drive(output_dir, drive_base_dir="/content/drive/MyDrive/GeorgianVerbSegmentation_Models"):
    """Copies the latest saved checkpoint from Drive back to local runtime if local is empty."""
    local_output_path = output_dir
    os.makedirs(local_output_path, exist_ok=True)

    # Check if local runtime already has checkpoints
    existing_local_ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if existing_local_ckpts:
        return  # Local already has it, no need to copy

    # Look for checkpoints in Google Drive
    drive_output_dir = os.path.join(
        drive_base_dir, os.path.basename(output_dir))
    if not os.path.exists(drive_output_dir):
        return

    drive_ckpts = [os.path.join(drive_output_dir, d) for d in os.listdir(drive_output_dir) if d.startswith(
        "checkpoint-") and os.path.isdir(os.path.join(drive_output_dir, d))]
    if not drive_ckpts:
        return

    # Find the latest checkpoint in Drive based on step number or modification time
    latest_drive_ckpt = max(drive_ckpts, key=os.path.getmtime)
    ckpt_name = os.path.basename(latest_drive_ckpt)

    local_target_ckpt = os.path.join(local_output_path, ckpt_name)

    print(
        f"🔄 [Drive Restore] Found checkpoint in Drive ({ckpt_name}). Copying to local runtime...")
    shutil.copytree(latest_drive_ckpt, local_target_ckpt, dirs_exist_ok=True)


def backup_latest_checkpoint_to_drive(output_dir, drive_base_dir="/content/drive/MyDrive/GeorgianVerbSegmentation_Models", keep_last_n=1):
    """Copies the latest checkpoint to Google Drive and cleans up older ones to save space."""
    if not os.path.exists("/content/drive"):
        return

    # 1. Find all local checkpoints
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith(
        "checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not checkpoints:
        return

    # Get the latest checkpoint based on modification time
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    ckpt_name = os.path.basename(latest_ckpt)

    drive_output_dir = os.path.join(
        drive_base_dir, os.path.basename(output_dir))
    target_dir = os.path.join(drive_output_dir, ckpt_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n☁️ [Drive Backup] Copying {ckpt_name} to Google Drive...")

    # Copy files
    for item in os.listdir(latest_ckpt):
        s_item = os.path.join(latest_ckpt, item)
        d_item = os.path.join(target_dir, item)
        if os.path.isdir(s_item):
            shutil.copytree(s_item, d_item, dirs_exist_ok=True)
        else:
            shutil.copy2(s_item, d_item)

    print(f"  ✓ Successfully backed up {ckpt_name} to Drive!")

    # 2. Cleanup older checkpoints in Google Drive to control file size
    if os.path.exists(drive_output_dir):
        drive_ckpts = [os.path.join(drive_output_dir, d) for d in os.listdir(drive_output_dir) if d.startswith(
            "checkpoint-") and os.path.isdir(os.path.join(drive_output_dir, d))]

        # Sort checkpoints by modification time (oldest first)
        drive_ckpts.sort(key=os.path.getmtime)

        # If we have more than keep_last_n, delete the oldest ones
        while len(drive_ckpts) > keep_last_n:
            old_ckpt = drive_ckpts.pop(0)
            print(
                f"🧹 [Drive Cleanup] Removing old checkpoint to save space: {os.path.basename(old_ckpt)}")
            shutil.rmtree(old_ckpt)


class DriveCheckpointCallback(TrainerCallback):
    """Custom Hugging Face callback to trigger a Drive backup after each epoch/save."""

    def on_save(self, args, state, control, **kwargs):
        backup_latest_checkpoint_to_drive(args.output_dir)


def split_data_by_verb(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """Splits dataset by verb_id / lemma to prevent data leakage."""
    unique_words = df["word_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_words)

    n_train = int(len(unique_words) * train_ratio)
    n_val = int(len(unique_words) * val_ratio)

    train_ids = set(unique_words[:n_train])
    val_ids = set(unique_words[n_train:n_train + n_val])
    test_ids = set(unique_words[n_train + n_val:])

    train_df = df[df["word_id"].isin(train_ids)].copy()
    val_df = df[df["word_id"].isin(val_ids)].copy()
    test_df = df[df["word_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df


def run_training(
    words_path: str = "words.tsv",
    forms_path: str = "forms.tsv",
    mode: str = "single",
    epochs: int = 5,
    batch_size: int = 16,
    max_samples: int = None,
    model_name: str = "google/byt5-small",
    smoke_test: bool = False
):
    """Encodes data, tokenizes, and fine-tunes Seq2Seq models."""
    start_time = time.time()
    print(f"--- Starting Seq2Seq Training (Mode: {mode}) ---")

    # Handle smoke test overrides
    if smoke_test:
        print("⚠️ [Smoke Test Mode Enabled]")
        save_strategy = "no"
        eval_strategy = "steps"   # Turn evaluation ON so we test compute_metrics!
        eval_steps = 10           # Run evaluation at step 10 and step 20
        logging_steps = 1
        max_steps = 20
        if max_samples is None:
            max_samples = 300
    else:
        eval_strategy = "epoch"
        eval_steps = None
        save_strategy = "epoch"
        logging_steps = 100
        max_steps = -1

    # 1. Load Data
    raw_df = load.load_and_merge_data(words_path, forms_path)
    if max_samples:
        raw_df = raw_df.head(max_samples)
        print(
            f"⚠️ Truncated dataset to {max_samples} samples for verification.")

    train_df, val_df, test_df = split_data_by_verb(raw_df)

    print(
        f"Train DF — Rows: {len(train_df)}, Unique words: {train_df['word_id'].nunique()}")
    print(
        f"Val DF   — Rows: {len(val_df)}, Unique words: {val_df['word_id'].nunique()}")
    print(
        f"Test DF  — Rows: {len(test_df)}, Unique words: {test_df['word_id'].nunique()}")

    # 2. Encode to Prompts
    train_encoded = seq_2_seq.encode(train_df, mode=mode)
    val_encoded = seq_2_seq.encode(val_df, mode=mode)
    test_encoded = seq_2_seq.encode(test_df, mode=mode)

    # 3. Tokenizer & Model setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenization function
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"], max_length=256, truncation=True)
        labels = tokenizer(
            text_target=examples["target_text"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Map datasets and strip raw text columns to prevent tensor collision errors
    train_dataset = Dataset.from_pandas(train_encoded).map(
        preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
    val_dataset = Dataset.from_pandas(val_encoded).map(
        preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
    test_dataset = Dataset.from_pandas(test_encoded).map(
        preprocess_function, batched=True, remove_columns=["input_text", "target_text"])

    # 4. Metrics Compute Function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace -100 with pad_token_id in BOTH preds and labels
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        return evaluate.evaluate_predictions(decoded_labels, decoded_preds)

    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./results_seq2seq_{mode}",
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        max_steps=max_steps,
        predict_with_generate=True,
        generation_max_length=128,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=logging_steps,
        # disable_tqdm=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if eval_strategy != "no" else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics if eval_strategy != "no" else None,
        callbacks=[DriveCheckpointCallback()]
    )

    # 6. Train & Evaluate
    # (Safely checks for checkpoints, skips for smoke tests)
    from transformers.trainer_utils import get_last_checkpoint

    restore_latest_checkpoint_from_drive(training_args.output_dir)

# Restore from Drive if local runtime was wiped out
    restore_latest_checkpoint_from_drive(training_args.output_dir)
    checkpoint_to_resume = None
    if not smoke_test:
        checkpoint_to_resume = get_last_checkpoint(training_args.output_dir)
        if checkpoint_to_resume:
            print(
                f"🔄 Resuming training from checkpoint: {checkpoint_to_resume}")
        else:
            print("🚀 No checkpoint found. Starting fresh training run.")
    else:
        print("🧪 Smoke test mode: Running fresh training (ignoring checkpoints).")

    # Inspect a single batch from the data collator to check if labels are all -100
    sample_batch = next(iter(trainer.get_train_dataloader()))
    print("Input IDs shape:", sample_batch["input_ids"].shape)
    print("Labels shape:", sample_batch["labels"].shape)
    # Should NOT be all -100
    print("Sample label tokens:", sample_batch["labels"][0][:20])

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    test_metrics = {}
    if eval_strategy != "no":
        print("\n--- Test Set Evaluation ---")
        test_results = trainer.predict(test_dataset)
        test_metrics = test_results.metrics

        preds = test_results.predictions
        labels = test_results.label_ids
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        input_texts = test_encoded["input_text"].tolist()

        evaluate.export_error_analysis(
            input_texts,
            decoded_labels,
            decoded_preds,
            output_path=f"error_analysis_seq2seq_{mode}.csv",
        )

        print("Test Metrics:", test_metrics)

    elapsed_time = time.time() - start_time

    metrics_dict = {
        "exact_match": test_metrics.get("test_exact_match", 0.0),
        "norm_edit_dist": test_metrics.get("test_norm_edit_dist", 0.0),
        "positional_slot_acc": test_metrics.get("test_positional_slot_acc", 0.0),
        "morpheme_f1": test_metrics.get("test_morpheme_f1", 0.0),
        "training_time_sec": round(elapsed_time, 1)
    }
    experiments.log_experiment(model_name, mode, metrics_dict)
    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words_path", type=str, default="words.tsv")
    parser.add_argument("--forms_path", type=str, default="forms.tsv")
    parser.add_argument("--mode", type=str,
                        choices=["single", "group6"], default="single")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="google/byt5-small")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a quick local/Colab check")
    args = parser.parse_args()

    run_training(
        words_path=args.words_path,
        forms_path=args.forms_path,
        mode=args.mode,
        epochs=args.epochs,
        max_samples=args.max_samples,
        model_name=args.model_name,
        smoke_test=args.smoke_test
    )
