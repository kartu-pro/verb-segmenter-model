"""
utils.py
--------
Shared utility functions for data splitting and Google Drive checkpoint management.
"""

import os
import shutil
import glob
import numpy as np
import pandas as pd
from transformers import TrainerCallback


def split_data_by_verb(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """Splits dataset by word_id / lemma to prevent data leakage."""
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


def restore_latest_checkpoint_from_drive(output_dir, drive_base_dir="/content/drive/MyDrive/GeorgianVerbSegmentation_Models"):
    """Copies the latest saved checkpoint from Drive back to local runtime if local is empty."""
    os.makedirs(output_dir, exist_ok=True)

    if glob.glob(os.path.join(output_dir, "checkpoint-*")):
        return  

    drive_output_dir = os.path.join(drive_base_dir, os.path.basename(output_dir))
    if not os.path.exists(drive_output_dir):
        return

    drive_ckpts = [os.path.join(drive_output_dir, d) for d in os.listdir(drive_output_dir) 
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(drive_output_dir, d))]
    if not drive_ckpts:
        return

    latest_drive_ckpt = max(drive_ckpts, key=os.path.getmtime)
    ckpt_name = os.path.basename(latest_drive_ckpt)
    local_target_ckpt = os.path.join(output_dir, ckpt_name)

    print(f"🔄 [Drive Restore] Found checkpoint ({ckpt_name}). Copying to local runtime...")
    shutil.copytree(latest_drive_ckpt, local_target_ckpt, dirs_exist_ok=True)


def backup_latest_checkpoint_to_drive(output_dir, drive_base_dir="/content/drive/MyDrive/GeorgianVerbSegmentation_Models", keep_last_n=1):
    """Copies the latest checkpoint to Google Drive and cleans up older ones."""
    if not os.path.exists("/content/drive"):
        return

    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not checkpoints:
        return

    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    ckpt_name = os.path.basename(latest_ckpt)

    drive_output_dir = os.path.join(drive_base_dir, os.path.basename(output_dir))
    target_dir = os.path.join(drive_output_dir, ckpt_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n☁️ [Drive Backup] Copying {ckpt_name} to Google Drive...")
    for item in os.listdir(latest_ckpt):
        s_item = os.path.join(latest_ckpt, item)
        d_item = os.path.join(target_dir, item)
        if os.path.isdir(s_item):
            shutil.copytree(s_item, d_item, dirs_exist_ok=True)
        else:
            shutil.copy2(s_item, d_item)

    # Cleanup older checkpoints in Drive
    if os.path.exists(drive_output_dir):
        drive_ckpts = [os.path.join(drive_output_dir, d) for d in os.listdir(drive_output_dir) 
                       if d.startswith("checkpoint-") and os.path.isdir(os.path.join(drive_output_dir, d))]
        drive_ckpts.sort(key=os.path.getmtime)
        while len(drive_ckpts) > keep_last_n:
            old_ckpt = drive_ckpts.pop(0)
            shutil.rmtree(old_ckpt)


class DriveCheckpointCallback(TrainerCallback):
    """Hugging Face callback to trigger a Drive backup after each save."""
    def on_save(self, args, state, control, **kwargs):
        backup_latest_checkpoint_to_drive(args.output_dir)