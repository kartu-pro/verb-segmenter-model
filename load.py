import json
import pandas as pd
from typing import Dict, Any

SLOT_KEYS = ["prefix", "person", "version", "root", "pfsf", "suffix"]


def parse_json_attr(val: Any) -> Dict[str, Any]:
    """Safely parse JSON attributes from a TSV cell."""
    if isinstance(val, dict):
        return val
    if pd.isna(val) or not val:
        return {}
    try:
        return json.loads(val)
    except Exception:
        return {}


def load_and_merge_data(words_path: str, forms_path: str) -> pd.DataFrame:
    """Loads TSVs, filters for verbs, and constructs standard target strings."""
    words_df = pd.read_csv(words_path, sep="\t")
    forms_df = pd.read_csv(forms_path, sep="\t")

    # Filter for verbs only
    words_df = words_df[words_df["pos"] == "verb"].copy()

    # Merge on word_id
    df = pd.merge(forms_df, words_df, on="word_id",
                  suffixes=("_form", "_word"))

    # Unpack JSON attributes
    df["word_attr"] = df["attributes_word"].apply(parse_json_attr)
    df["form_attr"] = df["attributes_form"].apply(parse_json_attr)

    # Metadata extraction
    df["screeve"] = df["form_attr"].apply(
        lambda d: d.get("screeve", "unknown"))
    df["subject"] = df["form_attr"].apply(
        lambda d: d.get("log_subj", "unknown"))
    df["verb_class"] = df["word_attr"].apply(
        lambda d: str(d.get("verb_class", "unknown")))
    df["lemma"] = df["lemma"].fillna("unknown")

    # Slot dictionary & standard segmented target
    df["slots"] = df["form_attr"].apply(
        lambda attr: {k: attr.get(k) for k in SLOT_KEYS if attr.get(k)}
    )
    df["segmented_target"] = df["form_attr"].apply(
        lambda attr: "-".join([attr.get(k, "") for k in SLOT_KEYS])
    )

    return df
