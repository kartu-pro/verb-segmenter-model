"""
save_artifacts.py
-----------------
Auto-discovers and backs up all pipeline artifacts to Google Drive 
without overwriting previous runs, keeping Seq2Seq backups lightweight 
by storing only trainer_state.json.
"""

import os
import glob
import shutil
from datetime import datetime


def save_to_google_drive(drive_base_dir="/content/drive/MyDrive/GeorgianVerbSegmentation_Backups"):
    """
    Finds all generated artifacts and copies them to a timestamped Google Drive folder.
    """
    # 1. Ensure Google Drive is mounted
    if not os.path.exists("/content/drive"):
        print("📁 Mounting Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')

    # 2. Create a unique timestamped run directory to prevent overwrites
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = os.path.join(drive_base_dir, f"run_{timestamp}")
    os.makedirs(target_dir, exist_ok=True)

    # 3. Define artifact patterns to glob
    patterns = [
        "experiment_results.json",
        "segmentation_benchmark_results.csv",
        "*.png",                      # Catches all plots
        "error_analysis_*.csv",       # Catches all error analysis files
        "results_tagger_*_history.json", # PyTorch tagger histories
    ]

    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern))

    copied_count = 0

    # Copy individual files
    for file_path in set(found_files):
        if os.path.isfile(file_path):
            shutil.copy(file_path, target_dir)
            print(f"  ✓ Copied file: {file_path}")
            copied_count += 1

    # 4. Lightweight Seq2Seq backup (only copy trainer_state.json)
    seq2seq_dirs = [d for d in glob.glob("results_seq2seq_*") if os.path.isdir(d)]
    for dir_path in seq2seq_dirs:
        state_file = os.path.join(dir_path, "trainer_state.json")
        if os.path.exists(state_file):
            dest_sub_dir = os.path.join(target_dir, os.path.basename(dir_path))
            os.makedirs(dest_sub_dir, exist_ok=True)
            shutil.copy(state_file, dest_sub_dir)
            print(f"  ✓ Copied lightweight state: {state_file}")
            copied_count += 1

    if copied_count > 0:
        print(f"\n🎉 Successfully backed up {copied_count} artifact group(s) to:\n{target_dir}")
    else:
        print("\n⚠️ No pipeline artifacts found in the current workspace to back up.")


if __name__ == "__main__":
    save_to_google_drive()