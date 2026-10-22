"""
train_seq2seq.py
----------------
Training pipeline for Sequence-to-Sequence models using HuggingFace Transformers & ByT5/T5.

Supports both 'single' word and 'group6' paradigm prompt modes.
Runs verb-grouped train/val/test splits to avoid data leakage using shared utilities.
"""

import time
import argparse
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

import load
import seq_2_seq
import evaluate
import experiments
from utils import (
    split_data_by_verb,
    restore_latest_checkpoint_from_drive,
    DriveCheckpointCallback
)


def _get_preprocess_function(tokenizer):
    """Returns a tokenization and preprocessing function for datasets."""
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"], max_length=256, truncation=True
        )
        labels = tokenizer(
            text_target=examples["target_text"], max_length=128, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function


def _get_compute_metrics(tokenizer):
    """Returns a metrics computation function for Seq2Seq evaluation."""
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        return evaluate.evaluate_predictions(decoded_labels, decoded_preds)
    return compute_metrics


def _prepare_test_dataset(words_path, forms_path, mode, tokenizer):
    """Shared helper: Loads data, extracts test split, encodes, and tokenizes into a Dataset."""
    raw_df = load.load_and_merge_data(words_path, forms_path)
    _, _, test_df = split_data_by_verb(raw_df)
    test_encoded = seq_2_seq.encode(test_df, mode=mode)
    
    preprocess_fn = _get_preprocess_function(tokenizer)
    test_dataset = Dataset.from_pandas(test_encoded).map(
        preprocess_fn, batched=True, remove_columns=["input_text", "target_text"]
    )
    return test_dataset, test_encoded


def _evaluate_and_log(trainer, test_dataset, test_encoded, tokenizer, start_time, model_name, mode):
    """Shared helper: Runs test prediction, extracts metrics, decodes, and logs via experiments module."""
    print("\n--- Test Set Evaluation ---")
    test_results = trainer.predict(test_dataset)
    test_metrics = test_results.metrics

    preds = np.where(test_results.predictions != -100, test_results.predictions, tokenizer.pad_token_id)
    labels = np.where(test_results.label_ids != -100, test_results.label_ids, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    input_texts = test_encoded["input_text"].tolist()

    elapsed_time = time.time() - start_time
    metrics_dict = {
        "exact_match": test_metrics.get("test_exact_match", 0.0),
        "norm_edit_dist": test_metrics.get("test_norm_edit_dist", 0.0),
        "positional_slot_acc": test_metrics.get("test_positional_slot_acc", 0.0),
        "morpheme_precision": test_metrics.get("test_morpheme_precision", 0.0),
        "morpheme_recall": test_metrics.get("test_morpheme_recall", 0.0),
        "morpheme_f1": test_metrics.get("test_morpheme_f1", 0.0),
        "training_time_sec": round(elapsed_time, 1)
    }

    # Centralized logging hub handles table appending and error analysis CSV export
    experiments.log_experiment(
        model_name=model_name,
        strategy_mode=mode,
        metrics=metrics_dict,
        error_inputs=input_texts,
        error_golds=decoded_labels,
        error_preds=decoded_preds,
        error_output_path=f"error_analysis_seq2seq_{mode}.csv"
    )
    print("Test Metrics:", test_metrics)
    return metrics_dict


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
        eval_strategy = "steps"
        eval_steps = 10
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

    # 1. Load & Split Data for Training
    raw_df = load.load_and_merge_data(words_path, forms_path)
    if max_samples:
        raw_df = raw_df.head(max_samples)
        print(f"⚠️ Truncated dataset to {max_samples} samples for verification.")

    train_df, val_df, test_df = split_data_by_verb(raw_df)

    print(f"Train DF — Rows: {len(train_df)}, Unique words: {train_df['word_id'].nunique()}")
    print(f"Val DF   — Rows: {len(val_df)}, Unique words: {val_df['word_id'].nunique()}")
    print(f"Test DF  — Rows: {len(test_df)}, Unique words: {test_df['word_id'].nunique()}")

    # 2. Encode to Prompts
    train_encoded = seq_2_seq.encode(train_df, mode=mode)
    val_encoded = seq_2_seq.encode(val_df, mode=mode)
    test_encoded = seq_2_seq.encode(test_df, mode=mode)

    # 3. Tokenizer & Model Setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    preprocess_fn = _get_preprocess_function(tokenizer)

    train_dataset = Dataset.from_pandas(train_encoded).map(
        preprocess_fn, batched=True, remove_columns=["input_text", "target_text"])
    val_dataset = Dataset.from_pandas(val_encoded).map(
        preprocess_fn, batched=True, remove_columns=["input_text", "target_text"])
    test_dataset = Dataset.from_pandas(test_encoded).map(
        preprocess_fn, batched=True, remove_columns=["input_text", "target_text"])

    # 4. Training Arguments
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
        report_to="none",
        metric_for_best_model="morpheme_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if eval_strategy != "no" else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=_get_compute_metrics(tokenizer) if eval_strategy != "no" else None,
        callbacks=[DriveCheckpointCallback()]
    )

    # 5. Restore & Checkpoint Resumption
    restore_latest_checkpoint_from_drive(training_args.output_dir)
    checkpoint_to_resume = None
    if not smoke_test:
        checkpoint_to_resume = get_last_checkpoint(training_args.output_dir)
        if checkpoint_to_resume:
            print(f"🔄 Resuming training from checkpoint: {checkpoint_to_resume}")
        else:
            print("🚀 No checkpoint found. Starting fresh training run.")
    else:
        print("🧪 Smoke test mode: Running fresh training (ignoring checkpoints).")

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # 6. Test Set Evaluation & Error Analysis
    if eval_strategy != "no":
        return _evaluate_and_log(trainer, test_dataset, test_encoded, tokenizer, start_time, model_name, mode)
    return {}


def evaluate_saved_checkpoint(
    mode: str = "single",
    model_name: str = "google/byt5-small",
    words_path: str = "words.tsv",
    forms_path: str = "forms.tsv"
):
    """Loads the latest checkpoint, runs test evaluation, exports error analysis, and logs results."""
    start_time = time.time()
    output_dir = f"./results_seq2seq_{mode}"

    restore_latest_checkpoint_from_drive(output_dir)
    checkpoint_path = get_last_checkpoint(output_dir)

    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found in {output_dir} or Google Drive.")

    print(f"📂 Loading checkpoint: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    # Reuse shared test dataset preparation helper
    test_dataset, test_encoded = _prepare_test_dataset(words_path, forms_path, mode, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=128,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=_get_compute_metrics(tokenizer)
    )

    print("\n--- Running Test Set Evaluation from Checkpoint ---")
    return _evaluate_and_log(trainer, test_dataset, test_encoded, tokenizer, start_time, model_name, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words_path", type=str, default="words.tsv")
    parser.add_argument("--forms_path", type=str, default="forms.tsv")
    parser.add_argument("--mode", type=str, choices=["single", "group6"], default="single")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="google/byt5-small")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick local/Colab check")
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