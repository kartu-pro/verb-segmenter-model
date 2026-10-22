"""
tagger_datasets.py
------------------
PyTorch Dataset and collation utilities for sequence tagging pipelines.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class GeorgianTaggingDataset(Dataset):
    """PyTorch Dataset wrapping encoded integer IDs."""

    def __init__(self, encoded_df: pd.DataFrame):
        if "token_ids" in encoded_df.columns:
            encoded_df = encoded_df[encoded_df["token_ids"].apply(
                lambda x: len(x) > 0)].reset_index(drop=True)
        elif "seq_len" in encoded_df.columns:
            encoded_df = encoded_df[encoded_df["seq_len"] > 0].reset_index(drop=True)

        self.data = encoded_df.to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "token_ids": torch.tensor(item["token_ids"], dtype=torch.long),
            "label_ids": torch.tensor(item["label_ids"], dtype=torch.long),
            "screeve_id": torch.tensor(item["screeve_id"], dtype=torch.long),
            "subject_id": torch.tensor(item["subject_id"], dtype=torch.long),
            "class_id": torch.tensor(item["class_id"], dtype=torch.long),
            "seq_len": item["seq_len"]
        }


def get_collate_fn(use_crf: bool):
    """Returns a collate function configured for standard CrossEntropy or CRF padding labels."""
    label_pad_value = 0 if use_crf else -100

    def pad_collate_fn(batch):
        token_ids = [b["token_ids"] for b in batch]
        label_ids = [b["label_ids"] for b in batch]

        padded_tokens = nn.utils.rnn.pad_sequence(
            token_ids, batch_first=True, padding_value=0)
        padded_labels = nn.utils.rnn.pad_sequence(
            label_ids, batch_first=True, padding_value=label_pad_value)

        screeve_ids = torch.stack([b["screeve_id"] for b in batch])
        subject_ids = torch.stack([b["subject_id"] for b in batch])
        class_ids = torch.stack([b["class_id"] for b in batch])

        return {
            "tokens": padded_tokens,
            "labels": padded_labels,
            "screeve_id": screeve_ids,
            "subject_id": subject_ids,
            "class_id": class_ids
        }
    return pad_collate_fn