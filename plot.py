"""
plot.py
-------
Visualization script for HuggingFace Seq2Seq training logs (trainer_state.json).
Supports inspecting different results folders and plotting training loss and evaluation metrics.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_cross_model_comparison(results_file="experiment_results.json"):
    """Generates a comparative bar chart across all logged model runs."""
    if not os.path.exists(results_file):
        print(f"❌ Error: {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    if not data:
        print("⚠️ No experiments logged yet.")
        return

    df = pd.DataFrame(data)
    df["Model Config"] = df["Model Architecture"] + \
        " (" + df["Strategy / Mode"] + ")"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Morpheme F1 & Exact Match Comparison
    df.plot(
        x="Model Config",
        y=["Morpheme F1 (%)", "Exact Match (%)"],
        kind="bar",
        ax=axes[0],
        colormap="viridis",
        rot=25,
    )
    axes[0].set_title("Performance Comparison", fontweight="bold")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", linestyle=":", alpha=0.7)

    # 2. Training Time Comparison
    df.plot(
        x="Model Config",
        y=["Training Time (s)"],
        kind="bar",
        ax=axes[1],
        color="salmon",
        rot=25,
    )
    axes[1].set_title("Training Efficiency", fontweight="bold")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(axis="y", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig("cross_model_comparison.png", dpi=300)
    print("✓ Cross-model comparison plot saved to 'cross_model_comparison.png'")
    plt.show()


def plot_training_results(results_dir: str, save_path: str = None):
    """Loads trainer_state.json from the specified results directory and generates diagnostic plots."""
    state_path = os.path.join(results_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print(
            f"❌ Error: Could not find 'trainer_state.json' in directory: {results_dir}")
        return

    with open(state_path, "r") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    if not logs:
        print(f"⚠️ Warning: 'log_history' is empty in {state_path}")
        return

    df = pd.DataFrame(logs)

    # Separate training loss logs and evaluation logs
    train_df = df.dropna(subset=["loss"])
    eval_df = df.dropna(subset=["eval_loss"])

    # Create a clean side-by-side plot layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Loss Curves ---
    if not train_df.empty:
        axes[0].plot(train_df["step"], train_df["loss"],
                     label="Training Loss", color="blue", alpha=0.6)
    if not eval_df.empty:
        axes[0].plot(eval_df["step"], eval_df["eval_loss"],
                     label="Validation Loss", color="orange", marker="o", linestyle="--")

    axes[0].set_title(
        f"Loss Curves\n({results_dir})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.7)

    # --- Plot 2: Evaluation Metrics ---
    # Filter out non-metric columns
    ignored_evals = {"eval_loss", "eval_runtime",
                     "eval_samples_per_second", "eval_steps_per_second", "eval_epoch"}
    metric_cols = [col for col in df.columns if col.startswith(
        "eval_") and col not in ignored_evals]

    if metric_cols and not eval_df.empty:
        for col in metric_cols:
            metric_name = col.replace("eval_", "")
            axes[1].plot(eval_df["step"], eval_df[col],
                         label=metric_name, marker="s", linestyle="-")
        axes[1].set_title(
            f"Evaluation Metrics\n({results_dir})", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, "No validation metrics logged yet\n(or eval strategy was set to 'no')",
                     horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, fontsize=10, color="gray")
        axes[1].set_title("Evaluation Metrics", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✓ Plot successfully saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Hugging Face Seq2Seq training logs.")
    parser.add_argument("--results_dir", type=str, default="./results_seq2seq_single",
                        help="Path to results directory containing trainer_state.json")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Optional file path to save the generated plot image")
    args = parser.parse_args()

    plot_training_results(args.results_dir, args.save_path)
