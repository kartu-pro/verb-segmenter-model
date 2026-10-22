"""
experiments.py (Updated)
------------------------
Handles experiment tracking, JSON/CSV result logging, summary tables, and reports.
"""

import os
import json
import pandas as pd
from IPython.display import display
import plot  # Import your plotting module

RESULTS_FILE = "experiment_results.json"


def init_tracking():
    """Clears or initializes the local results tracking file."""
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)
    print("Initialized experiment tracking container.")


def log_experiment(model_name: str, strategy_mode: str, metrics: dict):
    """Appends a completed experiment result to the semi-persistent JSON file."""
    row = {
        "Model Architecture": model_name,
        "Strategy / Mode": strategy_mode,
        "Exact Match (%)": metrics.get("exact_match", 0.0),
        "Norm Edit Dist": metrics.get("norm_edit_dist", 0.0),
        "Slot Accuracy (%)": metrics.get("positional_slot_acc", 0.0),
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


def generate_final_report():
    """
    Single method call to generate the full experiment suite report:
      1. Displays the highlighted summary table.
      2. Exports the comparison CSV.
      3. Renders and saves the cross-model performance comparison plots.
    """
    if not os.path.exists(RESULTS_FILE):
        print("❌ No experiment results found. Run some training cells first!")
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
        subset=["Exact Match (%)", "Morpheme F1 (%)", "Slot Accuracy (%)"], color="lightgreen"
    ).highlight_min(
        subset=["Norm Edit Dist", "Training Time (s)"], color="lightgreen"
    )
    display(styled_df)

    # Export for presentation/reports
    summary_df.to_csv("segmentation_benchmark_results.csv", index=False)
    print("\n✓ Saved comparison table to 'segmentation_benchmark_results.csv'")

    print("\n" + "=" * 80)
    print("GENERATING VISUAL REPORTS")
    print("=" * 80)
    plot.plot_cross_model_comparison(RESULTS_FILE)