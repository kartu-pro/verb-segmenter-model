"""
plot.py
-------
Comprehensive visualization suite for Georgian verb morpheme segmentation models.
Supports cross-model benchmarks (model-centric & metric-centric with model bars), 
rendered data tables with custom column widths and wrapped headers, combined PyTorch 
tagger loss curves, Hugging Face Seq2Seq training curves with scaled edit distance, 
multi-model slot error comparisons, and error analysis volume summaries.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from evaluate import parse_segmented_slots, SLOT_KEYS


# ==========================================
# MODULAR HELPER FUNCTIONS
# ==========================================

def plot_dataframe_table(results_file="experiment_results.json", save_path="results/plots/model_performance_table.png"):
    """Renders the pandas DataFrame results table into a clean image with a wider first column and wrapped headers."""
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

    # Define clean wrapped headers for the table
    wrapped_headers = [
        "Model Configuration", 
        "Morpheme\nF1 (%)", 
        "Exact\nMatch (%)", 
        "Slot\nAccuracy (%)", 
        "Morpheme\nPrecision (%)", 
        "Morpheme\nRecall (%)", 
        "Training\nTime (min)"
    ]

    display_cols = ["Model Config", "Morpheme F1 (%)", "Exact Match (%)", "Slot Accuracy (%)", "Morpheme Precision (%)", "Morpheme Recall (%)", "Training Time (s)"]
    existing_cols = [c for c in display_cols if c in df.columns]
    table_df = df[existing_cols]

    if "Training Time (s)" in table_df.columns:
        table_df["Training Time (min)"] = (table_df["Training Time (s)"] / 60.0).round(1)
        table_df = table_df.drop(columns=["Training Time (s)"])

    fig, ax = plt.subplots(figsize=(13, len(table_df) * 0.8 + 2.5))
    ax.axis('off')
    ax.axis('tight')

    # Custom column widths: make column 0 (Model name) wider, and metrics columns smaller
    num_cols = len(table_df.columns)
    col_widths = [0.32] + [0.11] * (num_cols - 1)

    table = ax.table(
        cellText=table_df.values,
        colLabels=wrapped_headers[:num_cols],
        colWidths=col_widths,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Style header row and table cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4c72b0')
            cell.set_text_props(weight='bold', color='white', ha='center', va='center')
        else:
            cell.set_facecolor('#f9f9f9' if row % 2 == 0 else '#ffffff')
            # Align model names to the left for readability, metrics centered
            if col == 0:
                cell.set_text_props(ha='left', weight='medium')

    plt.title("Model Benchmark Performance Summary Table", fontweight="bold", fontsize=14, pad=25)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Data table image saved to '{save_path}'")
    plt.close()


def plot_single_cross_model_comparison(results_file="experiment_results.json", save_path="results/plots/cross_model_comparison.png"):
    """Generates comparative multi-panel bar charts: model-centric (zoomed 80-100) and training duration in minutes."""
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 1. Core Metrics Comparison (Zoomed Y-axis: 80 to 100)
    metric_cols = [c for c in ["Morpheme F1 (%)", "Exact Match (%)", "Slot Accuracy (%)", "Morpheme Precision (%)", "Morpheme Recall (%)"] if c in df.columns]
    if metric_cols:
        df.plot(x="Model Config", y=metric_cols, kind="bar", ax=axes[0], colormap="viridis", rot=25)
        axes[0].set_title("Model Performance Metrics (Zoomed 80-100%)", fontweight="bold")
        axes[0].set_ylabel("Score (%)")
        axes[0].set_ylim(80, 100)
        axes[0].grid(axis="y", linestyle=":", alpha=0.7)
        axes[0].legend(loc="lower left")

    # 2. Training Time Comparison (Converted to Minutes with Data Labels)
    if "Training Time (s)" in df.columns:
        df["Training Time (min)"] = df["Training Time (s)"] / 60.0
        axes[1] = df.plot(x="Model Config", y=["Training Time (min)"], kind="bar", ax=axes[1], color="salmon", rot=25, legend=False)
        axes[1].set_title("Training Efficiency (Duration)", fontweight="bold")
        axes[1].set_ylabel("Duration (Minutes)")
        axes[1].grid(axis="y", linestyle=":", alpha=0.7)

        for container in axes[1].containers:
            axes[1].bar_label(container, fmt="%.1fm", padding=3, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✓ Cross-model model-centric plot saved to '{save_path}'")
    plt.close()


def plot_metric_centric_comparison(results_file="experiment_results.json", save_path="results/plots/metric_centric_comparison.png"):
    """Generates a cross-model comparison with metrics on X-axis and models as individual bars per metric."""
    if not os.path.exists(results_file):
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    if not data:
        return

    df = pd.DataFrame(data)
    df["Model Config"] = df["Model Architecture"] + " (" + df["Strategy / Mode"] + ")"
    
    metric_cols = [c for c in ["Morpheme F1 (%)", "Exact Match (%)", "Slot Accuracy (%)", "Morpheme Precision (%)", "Morpheme Recall (%)"] if c in df.columns]
    if not metric_cols:
        return

    df_pivoted = df.set_index("Model Config")[metric_cols].T

    fig, ax = plt.subplots(figsize=(12, 6))
    df_pivoted.plot(kind="bar", ax=ax, colormap="tab10", width=0.8, rot=15)

    ax.set_title("Model Performance Grouped by Metric", fontweight="bold", fontsize=14)
    ax.set_xlabel("Evaluation Metrics", fontweight="bold")
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_ylim(80, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.legend(title="Models", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metric-centric comparison plot saved to '{save_path}'")
    plt.close()


def plot_combined_tagger_histories(history_files: list, save_path="results/plots/tagger_training_loss_comparison.png"):
    """Plots all PyTorch sequence tagger training loss trajectories on a single graph with different series."""
    if not history_files:
        print("ℹ️ No PyTorch tagger history files found to plot.")
        return

    plt.figure(figsize=(10, 5))

    for hf in history_files:
        with open(hf, "r") as f:
            history = json.load(f)
        if not history.get("epoch"):
            continue
        
        model_label = os.path.basename(hf).replace("results_tagger_", "").replace("_history.json", "")
        plt.plot(history["epoch"], history["train_loss"], marker="o", linewidth=2, label=model_label)

    plt.title("PyTorch Tagger Training Loss Comparison", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✓ Combined tagger history plot saved to '{save_path}'")
    plt.close()


def plot_single_seq2seq_training(path_input: str):
    """Loads Hugging Face trainer_state.json and plots metrics, multiplying norm_edit_dist by 100 to fit scale."""
    if os.path.isfile(path_input):
        state_path = path_input
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

    # --- Plot 2: Evaluation Metrics (Scaled Norm Edit Dist) ---
    ignored_evals = {"eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "eval_epoch"}
    metric_cols = [col for col in df.columns if col.startswith("eval_") and col not in ignored_evals]

    if metric_cols and not eval_df.empty:
        for col in metric_cols:
            metric_name = col.replace("eval_", "")
            values = eval_df[col]
            
            if "norm_edit_dist" in col:
                values = values * 100.0
                metric_name = "norm_edit_dist (x100)"

            axes[1].plot(eval_df["step"], values, label=metric_name, marker="s", linestyle="-")

        axes[1].set_title(f"Seq2Seq Evaluation Metrics (Scaled)\n({identifier})", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Score / Scaled Value")
        axes[1].set_ylim(0, 100)
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        axes[1].grid(True, linestyle=":", alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, "No validation metrics logged yet", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, fontsize=10, color="gray")
        axes[1].set_title("Evaluation Metrics", fontsize=12, fontweight="bold")

    plt.tight_layout()
    
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{identifier}_training_plot.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Seq2Seq training results plot saved to '{save_path}'")
    plt.close()


def plot_single_error_summary(save_path="results/plots/error_analysis_summary.png"):
    """Plots comparative total error volume with slanted X-axis labels to prevent overlap."""
    error_files = glob.glob("results/raw/error_analysis_*.csv")
    if not error_files:
        print("ℹ️ No error analysis CSV files found to plot.")
        return

    error_counts = {}
    for ef in error_files:
        df = pd.read_csv(ef)
        model_label = os.path.basename(ef).replace("error_analysis_", "").replace(".csv", "")
        error_counts[model_label] = len(df)

    plt.figure(figsize=(10, 5))
    models = list(error_counts.keys())
    counts = list(error_counts.values())

    bars = plt.bar(models, counts, color="indianred", width=0.5)
    plt.title("Error Volume Comparison (Failed Predictions)", fontweight="bold")
    plt.xlabel("Model Variant / Strategy")
    plt.ylabel("Number of Errors")
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✓ Error analysis summary plot saved to '{save_path}'")
    plt.close()


def plot_combined_slot_error_breakdown(error_csv_files: list, save_path="results/plots/slot_errors_comparison.png"):
    """Combines slot-level error breakdowns into a single grouped bar chart across all models."""
    if not error_csv_files:
        print("ℹ️ No error analysis files found for slot breakdowns.")
        return

    slot_data = {slot: {} for slot in SLOT_KEYS}

    for ef in error_csv_files:
        df = pd.read_csv(ef)
        if df.empty or "Gold Segmentation" not in df.columns or "Predicted Segmentation" not in df.columns:
            continue
        
        model_label = os.path.basename(ef).replace("error_analysis_", "").replace(".csv", "")
        slot_error_counts = {slot: 0 for slot in SLOT_KEYS}

        for _, row in df.iterrows():
            gold_slots = parse_segmented_slots(str(row["Gold Segmentation"]))
            pred_slots = parse_segmented_slots(str(row["Predicted Segmentation"]))
            
            for idx, slot_name in enumerate(SLOT_KEYS):
                if gold_slots[idx] != pred_slots[idx]:
                    slot_error_counts[slot_name] += 1
        
        for slot in SLOT_KEYS:
            slot_data[slot][model_label] = slot_error_counts[slot]

    df_slots = pd.DataFrame(slot_data).T

    plt.figure(figsize=(12, 6))
    df_slots.plot(kind="bar", figsize=(12, 6), width=0.8, rot=0)
    
    plt.title("Slot-Level Error Breakdown Across All Models", fontweight="bold")
    plt.xlabel("Morpheme Positional Slots")
    plt.ylabel("Mismatched Count")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend(title="Models", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined slot error breakdown plot saved to '{save_path}'")
    plt.close()


# ==========================================
# AUTOMATED DISCOVERY & EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUTO-DISCOVERING ARTIFACTS AND GENERATING RELEVANT PLOTS")
    print("=" * 60)

    if os.path.exists("results/combined_experiment_results.json"):
        print("\n[1/6] Generating Cross-Model Benchmark Visualizations & Table Image...")
        plot_dataframe_table("results/combined_experiment_results.json")
        plot_single_cross_model_comparison("results/combined_experiment_results.json")
        plot_metric_centric_comparison("results/combined_experiment_results.json")
    else:
        print("\n[1/6] Skipping cross-model comparisons.")

    print("\n[2/6] Generating Error Analysis Summary Chart...")
    plot_single_error_summary()

    error_files = glob.glob("results/raw/error_analysis_*.csv")
    if error_files:
        print(f"\n[3/6] Generating Combined Slot-Level Error Breakdown for {len(error_files)} file(s)...")
        plot_combined_slot_error_breakdown(error_files)
    else:
        print("\n[3/6] No error analysis files found.")

    tagger_histories = glob.glob("results/raw/results_tagger_*_history.json")
    if tagger_histories:
        print(f"\n[4/6] Found {len(tagger_histories)} PyTorch Tagger history file(s). Plotting combined losses...")
        plot_combined_tagger_histories(tagger_histories)
    else:
        print("\n[4/6] No PyTorch tagger history files found.")

    if os.path.exists("results/raw/trainer_state_group6.json"):
        print(f"\n[5/6] Found trainer_state_group6.json. Plotting scaled metrics...")
        plot_single_seq2seq_training("results/raw/trainer_state_group6.json")

    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE!")
    print("=" * 60)