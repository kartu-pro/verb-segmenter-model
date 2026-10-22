"""
save_models.py
--------------
Backs up all 4 trained models (Seq2Seq: single, group6 & Tagger: bio, boundaries) 
to Google Drive. Skips gracefully if model files are not present.
"""

import os
import glob
import shutil


def save_models_to_drive(
    drive_base_dir="/content/drive/MyDrive/GeorgianVerbSegmentation_Models",
):
    """Finds all 4 trained model artifacts and backs them up to Google Drive."""
    if not os.path.exists("/content/drive"):
        print("📁 Mounting Google Drive...")
        from google.colab import drive

        drive.mount("/content/drive")

    os.makedirs(drive_base_dir, exist_ok=True)

    saved_count = 0

    # 1. Back up Seq2Seq Models (single, group6)
    seq2seq_dirs = [d for d in glob.glob("results_seq2seq_*") if os.path.isdir(d)]
    for dir_path in seq2seq_dirs:
        model_name = os.path.basename(dir_path)  # e.g., results_seq2seq_single
        target_sub_dir = os.path.join(drive_base_dir, model_name)

        # Verify weights exist inside
        has_weights = any(
            os.path.exists(os.path.join(dir_path, f))
            for f in ["model.safetensors", "pytorch_model.bin"]
        )
        if not has_weights:
            print(f"⏩ Skipping {model_name}: No model weights found inside.")
            continue

        print(f"📦 Backing up Seq2Seq model ({model_name}) (~1.2 GB)...")
        if os.path.exists(target_sub_dir):
            shutil.rmtree(target_sub_dir)
        shutil.copytree(dir_path, target_sub_dir)
        print(f"  ✓ Saved to: {target_sub_dir}")
        saved_count += 1

    # 2. Back up PyTorch Tagger Models (bio, boundaries)
    tagger_files = glob.glob("model_tagger_*.pt")
    for file_path in tagger_files:
        file_name = os.path.basename(file_path)  # e.g., model_tagger_bio.pt
        target_file_path = os.path.join(drive_base_dir, file_name)

        print(f"📦 Backing up Tagger model ({file_name})...")
        shutil.copy(file_path, target_file_path)
        print(f"  ✓ Saved to: {target_file_path}")
        saved_count += 1

    if saved_count > 0:
        print(
            f"\n🎉 Successfully backed up {saved_count} model(s) to Google Drive!"
        )
    else:
        print(
            "\n⚠️ No trained models (Seq2Seq or Taggers) were found to back up."
        )


if __name__ == "__main__":
    save_models_to_drive()