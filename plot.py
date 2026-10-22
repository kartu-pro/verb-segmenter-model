"""
plot.py
-------
Comprehensive visualization suite for Georgian verb morpheme segmentation models.
Supports cross-model benchmarks, custom PyTorch tagger histories, Hugging Face 
Seq2Seq training curves, and error analysis volume comparisons.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt


def plot_cross_model_comparison(results_file="experiment_results.json"):
    """
    Generates a comparative multi-panel bar chart across all logged model runs.
    
    What it shows:
      - Panel 1: Performance metrics comparison (%) including Morpheme F1, Exact Match, 
        Slot Accuracy, Precision, and Recall across all evaluated architectures.
      - Panel 2: Training efficiency comparison showing total training duration in seconds.
    """
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
    save_path = "cross_model_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"✓ Cross-model comparison plot saved to '{save_path}'")
    plt.close()


def plot_tagger_history(history_file: str):
    """
    Generates training trajectory plots for custom PyTorch sequence tagger runs.
    
    What it shows:
      - Training loss progression across epochs from `results_tagger_*_history.json`.
    """
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

    save_path = history_file.replace(".json", "_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Tagger history plot saved to '{save_path}'")
    plt.close()


def plot_training_results(results_dir: str):
    """
    Loads Hugging Face trainer_state.json from a Seq2Seq results directory and generates diagnostic plots.
    
    What it shows:
      - Panel 1: Training loss vs. Validation loss over optimization steps.
      - Panel 2: Evaluation metrics trajectories (e.g., exact match, F1, edit distance) logged during training.
    """
    state_path = os.path.join(results_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"❌ Error: Could not find 'trainer_state.json' in directory: {results_dir}")
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

    axes[0].set_title(f"Seq2Seq Loss Curves\n({results_dir})", fontsize=12, fontweight="bold")
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
        axes[1].set_title(f"Seq2Seq Evaluation Metrics\n({results_dir})", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, "No validation metrics logged yet\n(or eval strategy was set to 'no')",
                     horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, fontsize=10, color="gray")
        axes[1].set_title("Evaluation Metrics", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_path = f"{results_dir}_training_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"✓ Seq2Seq training results plot saved to '{save_path}'")
    plt.close()


def plot_error_analysis_summary():
    """
    Scans all error analysis CSV files in the workspace and plots comparative error volumes.
    
    What it shows:
      - A comparative bar chart showing the total count of failed/mispredicted test samples 
        logged across each error analysis file (`error_analysis_*.csv`).
    """
    error_files = glob.glob("error_analysis_*.csv")
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
    save_path = "error_analysis_summary.png"
    plt.savefig(save_path, dpi=300)
    print(f"✓ Error analysis summary plot saved to '{save_path}'")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("AUTO-DISCOVERING ARTIFACTS AND GENERATING RELEVANT PLOTS")
    print("=" * 60)

    # 1. Master benchmark comparison from experiment_results.json
    if os.path.exists("experiment_results.json"):
        print("\n[1/4] Generating Cross-Model Benchmark Comparison...")
        plot_cross_model_comparison("experiment_results.json")
    else:
        print("\n[1/4] Skipping cross-model comparison ('experiment_results.json' not found).")

    # 2. Error volume comparison from error_analysis_*.csv files
    print("\n[2/4] Generating Error Analysis Summary Chart...")
    plot_error_analysis_summary()

    # 3. Custom PyTorch Tagger training curves from history json files
    tagger_histories = glob.glob("results_tagger_*_history.json")
    if tagger_histories:
        print(f"\n[3/4] Found {len(tagger_histories)} PyTorch Tagger history file(s). Plotting...")
        for th in tagger_histories:
            plot_tagger_history(th)
    else:
        print("\n[3/4] No PyTorch tagger history files found.")

    # 4. Hugging Face Seq2Seq loss & evaluation curves from trainer states
    seq2seq_dirs = [d for d in glob.glob("results_seq2seq_*") if os.path.isdir(d)]
    if seq2seq_dirs:
        print(f"\n[4/4] Found {len(seq2seq_dirs)} Seq2Seq result directory/directories. Plotting...")
        for sd in seq2seq_dirs:
            if os.path.exists(os.path.join(sd, "trainer_state.json")):
                plot_training_results(sd)
    else:
        print("\n[4/4] No Seq2Seq result directories found.")

    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE!")
    print("=" * 60)