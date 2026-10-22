"""
experiments.py
--------------
Centralized experiment tracking, result logging, history saving, error analysis export, and reporting.
"""

import os
import json
import pandas as pd
from IPython.display import display
import plot
import evaluate

RESULTS_FILE = "experiment_results.json"


def init_tracking():
    """Clears or initializes the local results tracking file."""
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)
    print("Initialized experiment tracking container.")


def log_experiment(
    model_name: str,
    strategy_mode: str,
    metrics: dict,
    history: dict = None,
    error_inputs: list = None,
    error_golds: list = None,
    error_preds: list = None,
    error_output_path: str = None
):
    """
    Unified logging hub that:
      1. Appends metrics to the main experiment JSON table.
      2. Saves training history JSON curves (if provided).
      3. Automatically exports error analysis CSVs (if prediction lists are provided).
    """
    # 1. Log to main experiment summary table
    row = {
        "Model Architecture": model_name,
        "Strategy / Mode": strategy_mode,
        "Exact Match (%)": metrics.get("exact_match", 0.0),
        "Norm Edit Dist": metrics.get("norm_edit_dist", 0.0),
        "Slot Accuracy (%)": metrics.get("positional_slot_acc", 0.0),
        "Morpheme Precision (%)": metrics.get("morpheme_precision", 0.0),
        "Morpheme Recall (%)": metrics.get("morpheme_recall", 0.0),
        "Morpheme F1 (%)": metrics.get("morpheme_f1", 0.0),
        "Training Time (s)": metrics.get("training_time_sec", 0.0)
    }

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(row)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n[Logged Successfully] {model_name} ({strategy_mode}) | F1: {row['Morpheme F1 (%)']}%")

    # 2. Save Training History JSON (if provided)
    if history is not None:
        clean_model_tag = model_name.lower().replace("/", "_").replace("-", "_")
        history_path = f"./results_{clean_model_tag}_{strategy_mode}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        print(f"  ✓ Saved training history to '{history_path}'")

    # 3. Export Error Analysis CSV (if prediction components are provided)
    if error_inputs and error_golds and error_preds and error_output_path:
        evaluate.export_error_analysis(
            error_inputs,
            error_golds,
            error_preds,
            output_path=error_output_path
        )


def generate_final_report():
    """Generates the full summary table, CSV export, and cross-model plots."""
    if not os.path.exists(RESULTS_FILE):
        print("❌ No experiment results found.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    if not data:
        print("⚠️ Experiment log is empty.")
        return

    summary_df = pd.DataFrame(data)

    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print("=" * 80)

    styled_df = summary_df.style.highlight_max(
        subset=[
            "Exact Match (%)",
            "Slot Accuracy (%)",
            "Morpheme Precision (%)",
            "Morpheme Recall (%)",
            "Morpheme F1 (%)"
        ],
        color="lightgreen"
    ).highlight_min(
        subset=["Norm Edit Dist", "Training Time (s)"], color="lightgreen"
    )
    display(styled_df)

    summary_df.to_csv("segmentation_benchmark_results.csv", index=False)
    print("\n✓ Saved comparison table to 'segmentation_benchmark_results.csv'")

    print("\n" + "=" * 80)
    print("GENERATING VISUAL REPORTS")
    print("=" * 80)
    plot.plot_cross_model_comparison(RESULTS_FILE)