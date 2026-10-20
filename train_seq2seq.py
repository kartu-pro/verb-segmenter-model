"""
train_seq2seq.py
----------------
Training pipeline for Sequence-to-Sequence models using HuggingFace Transformers & ByT5.

Supports both 'single' word and 'group6' paradigm prompt modes.
Runs verb-grouped train/val/test splits to avoid data leakage.
"""

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


def run_training(words_path: str, forms_path: str, mode: str = "single", epochs: int = 5, batch_size: int = 16):
    """Encodes data, tokenizes, and fine-tunes ByT5."""
    print(f"--- Starting Seq2Seq Training (Mode: {mode}) ---")

    # 1. Load Data
    raw_df = load.load_and_merge_data(words_path, forms_path)
    train_df, val_df, test_df = split_data_by_verb(raw_df)

    # 2. Encode to Prompts
    train_encoded = seq_2_seq.encode(train_df, mode=mode)
    val_encoded = seq_2_seq.encode(val_df, mode=mode)
    test_encoded = seq_2_seq.encode(test_df, mode=mode)

    # 3. Tokenizer & Model setup (ByT5 operates natively on bytes/characters)
    model_name = "google/byt5-small"
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

    train_dataset = Dataset.from_pandas(train_encoded).map(
        preprocess_function, batched=True)
    val_dataset = Dataset.from_pandas(val_encoded).map(
        preprocess_function, batched=True)
    test_dataset = Dataset.from_pandas(test_encoded).map(
        preprocess_function, batched=True)

    # 4. Metrics Compute Function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        return evaluate.evaluate_predictions(decoded_labels, decoded_preds)

    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./results_seq2seq_{mode}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics
    )

    # 6. Train & Evaluate
    trainer.train()

    print("\n--- Test Set Evaluation ---")
    test_results = trainer.predict(test_dataset)
    print("Test Metrics:", test_results.metrics)

    # Extract just the core metric floats to match your logger
    return {
        "exact_match": test_results.metrics.get("test_exact_match", 0.0),
        "norm_edit_dist": test_results.metrics.get("test_norm_edit_dist", 0.0),
        "positional_slot_acc": test_results.metrics.get("test_positional_slot_acc", 0.0),
        "morpheme_f1": test_results.metrics.get("test_morpheme_f1", 0.0)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words_path", type=str, default="words.tsv")
    parser.add_argument("--forms_path", type=str, default="forms.tsv")
    parser.add_argument("--mode", type=str,
                        choices=["single", "group6"], default="single")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    run_training(args.words_path, args.forms_path,
                 mode=args.mode, epochs=args.epochs)
