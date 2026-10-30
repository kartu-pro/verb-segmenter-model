"""
plot.py
-------
Comprehensive visualization suite for Georgian verb morpheme segmentation models.
Supports cross-model benchmarks, custom PyTorch tagger histories, Hugging Face 
Seq2Seq training curves, slot-level error breakdowns, and error analysis volume comparisons.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from evaluate import parse_segmented_slots, SLOT_KEYS


# ==========================================
# MODULAR HELPER FUNCTIONS (One plot per function)
# ==========================================

def plot_single_cross_model_comparison(results_file="experiment_results.json", save_path="results/plots/cross_model_comparison.png"):
    """Helper function: Generates a comparative multi-panel bar chart across all logged model runs."""
    if not os.path.exists(results_file):
        print(f"❌ Error: {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    if not data:
        print("⚠️ No experiments logged yet.")
        return

    df = pd.DataFrame(data)
    df["Model Config"] = df["Model Architecture"] + " (" + df["Strategy / Mode"] + ")"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Core Metrics Comparison
    metric_cols = [c for c in ["Morpheme F1 (%)", "Exact Match (%)", "Slot Accuracy (%)", "Morpheme Precision (%)", "Morpheme Recall (%)"] if c in df.columns]
    if metric_cols:
        df.plot(x="Model Config", y=metric_cols, kind="bar", ax=axes[0], colormap="viridis", rot=25)
        axes[0].set_title("Model Performance Metrics", fontweight="bold")
        axes[0].set_ylabel("Score (%)")
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis="y", linestyle=":", alpha=0.7)

    # 2. Training Time Comparison
    if "Training Time (s)" in df.columns:
        df.plot(x="Model Config", y=["Training Time (s)"], kind="bar", ax=axes[1], color="salmon", rot=25)
        axes[1].set_title("Training Efficiency (Duration)", fontweight="bold")
        axes[1].set_ylabel("Seconds")
        axes[1].grid(axis="y", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Cross-model comparison plot saved to '{save_path}'")
    plt.close()


def plot_single_tagger_history(history_file: str):
    """Helper function: Generates training trajectory loss plot for a custom PyTorch sequence tagger run."""
    if not os.path.exists(history_file):
        print(f"❌ Error: {history_file} not found.")
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    if not history.get("epoch"):
        print(f"⚠️ Warning: Empty history in {history_file}")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], marker="o", color="blue", label="Train Loss")
    plt.title(f"PyTorch Tagger Training Loss\n({os.path.basename(history_file)})", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    filename = os.path.basename(history_file)
    name_without_ext = filename.replace(".json", "")
    save_filename = f"{name_without_ext}_plot.png"
    save_path = os.path.join("results/plots", save_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Tagger history plot saved to '{save_path}'")
    plt.close()


def plot_single_seq2seq_training(path_input: str):
    """Helper function: Loads Hugging Face trainer_state.json (direct file or directory) and plots loss/metrics."""
    # Handle whether a direct file path or a directory was passed
    if os.path.isfile(path_input):
        state_path = path_input
        # Extract name like 'trainer_state_group6'
        identifier = os.path.splitext(os.path.basename(state_path))[0]
    else:
        state_path = os.path.join(path_input, "trainer_state.json")
        identifier = os.path.basename(path_input)

    if not os.path.exists(state_path):
        print(f"❌ Error: Could not find 'trainer_state.json' at: {state_path}")
        return

    with open(state_path, "r") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    if not logs:
        print(f"⚠️ Warning: 'log_history' is empty in {state_path}")
        return

    df = pd.DataFrame(logs)
    train_df = df.dropna(subset=["loss"])
    eval_df = df.dropna(subset=["eval_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Loss Curves ---
    if not train_df.empty:
        axes[0].plot(train_df["step"], train_df["loss"], label="Training Loss", color="blue", alpha=0.6)
    if not eval_df.empty:
        axes[0].plot(eval_df["step"], eval_df["eval_loss"], label="Validation Loss", color="orange", marker="o", linestyle="--")

    axes[0].set_title(f"Seq2Seq Loss Curves\n({identifier})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.7)

    # --- Plot 2: Evaluation Metrics ---
    ignored_evals = {"eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "eval_epoch"}
    metric_cols = [col for col in df.columns if col.startswith("eval_") and col not in ignored_evals]

    if metric_cols and not eval_df.empty:
        for col in metric_cols:
            metric_name = col.replace("eval_", "")
            axes[1].plot(eval_df["step"], eval_df[col], label=metric_name, marker="s", linestyle="-")
        axes[1].set_title(f"Seq2Seq Evaluation Metrics\n({identifier})", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, "No validation metrics logged yet", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, fontsize=10, color="gray")
        axes[1].set_title("Evaluation Metrics", fontsize=12, fontweight="bold")

    plt.tight_layout()
    
    # --- SAVE PATH ---
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{identifier}_training_plot.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"✓ Seq2Seq training results plot saved to '{save_path}'")
    plt.close()


def plot_single_error_summary(save_path="results/plots/error_analysis_summary.png"):
    """Helper function: Plots comparative total error volume across all error analysis CSV files."""
    error_files = glob.glob("results/raw/error_analysis_*.csv")
    if not error_files:
        print("ℹ️ No error analysis CSV files found to plot.")
        return

    error_counts = {}
    for ef in error_files:
        df = pd.read_csv(ef)
        model_label = ef.replace("error_analysis_", "").replace(".csv", "")
        error_counts[model_label] = len(df)

    plt.figure(figsize=(8, 4))
    models = list(error_counts.keys())
    counts = list(error_counts.values())

    bars = plt.bar(models, counts, color="indianred", width=0.5)
    plt.title("Error Volume Comparison (Failed Predictions)", fontweight="bold")
    plt.xlabel("Model Variant / Strategy")
    plt.ylabel("Number of Errors")
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Error analysis summary plot saved to '{save_path}'")
    plt.close()


def plot_slot_error_breakdown(error_csv_path: str):
    """
    NEW IDEA HELPER: Scans a specific error analysis CSV file and counts 
    which positional slots fail most frequently (Prefix, Person, Version, Root, PFSF, Suffix).
    """
    if not os.path.exists(error_csv_path):
        print(f"❌ Error: {error_csv_path} not found.")
        return

    df = pd.read_csv(error_csv_path)
    if df.empty or "Gold Segmentation" not in df.columns or "Predicted Segmentation" not in df.columns:
        print(f"⚠️ Invalid or empty error analysis file: {error_csv_path}")
        return

    slot_error_counts = {slot: 0 for slot in SLOT_KEYS}

    for _, row in df.iterrows():
        gold_slots = parse_segmented_slots(str(row["Gold Segmentation"]))
        pred_slots = parse_segmented_slots(str(row["Predicted Segmentation"]))
        
        for idx, slot_name in enumerate(SLOT_KEYS):
            if gold_slots[idx] != pred_slots[idx]:
                slot_error_counts[slot_name] += 1

    plt.figure(figsize=(8, 4))
    slots = list(slot_error_counts.keys())
    errors = list(slot_error_counts.values())

    bars = plt.bar(slots, errors, color="purple", alpha=0.7, width=0.5)
    model_tag = error_csv_path.replace("error_analysis_", "").replace(".csv", "")
    plt.title(f"Slot-Level Error Breakdown\n({model_tag})", fontweight="bold")
    plt.xlabel("Morpheme Positional Slots")
    plt.ylabel("Mismatched Count")
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    base_name = os.path.basename(error_csv_path)
    clean_name = base_name.replace("error_analysis_", "slot_errors_").replace(".csv", ".png")
    
    save_path = os.path.join("results/plots", clean_name)
    plt.savefig(save_path, dpi=300)
    print(f"✓ Slot error breakdown plot saved to '{save_path}'")
    plt.close()


# ==========================================
# AUTOMATED DISCOVERY & EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUTO-DISCOVERING ARTIFACTS AND GENERATING RELEVANT PLOTS")
    print("=" * 60)

    # 1. Master benchmark comparison

    if os.path.exists("results/combined_experiment_results.json"):
        print("\n[1/5] Generating Cross-Model Benchmark Comparison...")
        plot_single_cross_model_comparison("results/combined_experiment_results.json")
    else:
        print("\n[1/5] Skipping cross-model comparison ('experiment_results.json' not found).")

    # 2. Error volume comparison
    print("\n[2/5] Generating Error Analysis Summary Chart...")
    plot_single_error_summary()

    # 3. New Slot-Level Error Breakdown for each error CSV
    error_files = glob.glob("results/raw/error_analysis_*.csv")
    if error_files:
        print(f"\n[3/5] Generating Slot-Level Error Breakdowns for {len(error_files)} error file(s)...")
        for ef in error_files:
            plot_slot_error_breakdown(ef)
    else:
        print("\n[3/5] No error analysis files found for slot breakdowns.")

    # 4. Custom PyTorch Tagger training curves
    tagger_histories = glob.glob("results/raw/results_tagger_*_history.json")
    if tagger_histories:
        print(f"\n[4/5] Found {len(tagger_histories)} PyTorch Tagger history file(s). Plotting...")
        for th in tagger_histories:
            plot_single_tagger_history(th)
    else:
        print("\n[4/5] No PyTorch tagger history files found.")

    # 5. Hugging Face Seq2Seq loss & evaluation curves
    print(f"\n[5/5] Found trainer_state_group6.json. Plotting...")
    plot_single_seq2seq_training("results/raw/trainer_state_group6.json")

    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE!")
    print("=" * 60)
