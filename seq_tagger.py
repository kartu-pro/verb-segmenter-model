import pandas as pd
from typing import List, Tuple, Dict, Any
from load import SLOT_KEYS

DELIMITER_MAP = {0: "", 1: "-", 2: "--", 3: "---"}

# ==========================================
# 1. VOCABULARY BUILDER & METADATA MAPPING
# ==========================================


def build_vocabularies(df: pd.DataFrame, strategy: str = "boundaries") -> Dict[str, Dict[str, int]]:
    """
    Builds string-to-integer mappings for characters, metadata, and target labels.

    Args:
        df: Input DataFrame from load.load_and_merge_data().
        strategy: 'boundaries' or 'bio'.

    Returns:
        Dict containing sub-dictionaries:
          - 'char2id': Vocabulary for character tokens.
          - 'screeve2id': Mappings for screeve categories.
          - 'subject2id': Mappings for subject categories.
          - 'class2id': Mappings for verb class categories.
          - 'label2id': Mappings for target boundary or BIO labels.

    Example Output:
        {
            "char2id": {"<PAD>": 0, "<UNK>": 1, "^": 2, "ა": 3, "ბ": 4, ...},
            "screeve2id": {"imperfect": 0, "present": 1, ...},
            "subject2id": {"1s": 0, "2s": 1, ...},
            "class2id": {"1": 0, "2": 1, ...},
            "label2id": {0: 0, 1: 1, 2: 2, 3: 3}
        }
    """
    all_chars = set("^")
    for form in df["form"].dropna():
        all_chars.update(list(form))

    char2id = {"<PAD>": 0, "<UNK>": 1}
    for idx, char in enumerate(sorted(all_chars), start=2):
        char2id[char] = idx

    screeve2id = {val: idx for idx, val in enumerate(
        sorted(df["screeve"].unique()))}
    subject2id = {val: idx for idx, val in enumerate(
        sorted(df["subject"].unique()))}
    class2id = {val: idx for idx, val in enumerate(
        sorted(df["verb_class"].unique()))}

    if strategy == "boundaries":
        label2id = {0: 0, 1: 1, 2: 2, 3: 3}
    elif strategy == "bio":
        all_tags = {"O"}
        for k in SLOT_KEYS:
            all_tags.add(f"B-{k.upper()}")
            all_tags.add(f"I-{k.upper()}")
        label2id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    else:
        raise ValueError("strategy must be 'boundaries' or 'bio'")

    return {
        "char2id": char2id,
        "screeve2id": screeve2id,
        "subject2id": subject2id,
        "class2id": class2id,
        "label2id": label2id
    }


# ==========================================
# 2. BIO SLOT TAGGING (STRATEGY: BIO)
# ==========================================

def encode_bio(attr_dict: dict) -> Tuple[List[str], List[str]]:
    """
    Encodes surface characters into BIO slot tags for existing morphemes.

    Args:
        attr_dict: Form attribute dictionary containing slot values.

    Example Input:
        attr_dict = {"version": "ა", "root": "კეთ", "pfsf": "ებ", "suffix": "ს"}

    Example Output:
        tokens = ['ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'ს']
        labels = ['B-VERSION', 'B-ROOT', 'I-ROOT', 'I-ROOT', 'B-PFSF', 'I-PFSF', 'B-SUFFIX']
    """
    chars, labels = [], []
    for key in SLOT_KEYS:
        val = attr_dict.get(key)
        if val:
            for idx, char in enumerate(val):
                tag = f"B-{key.upper()}" if idx == 0 else f"I-{key.upper()}"
                chars.append(char)
                labels.append(tag)
    return chars, labels


def decode_bio(tokens: List[str], predicted_bio_tags: List[str]) -> str:
    """
    Decodes tokens and predicted BIO tags into the 6-slot segmented template.
    Missing slots default to empty strings, generating exact hyphen counts via joining.

    Args:
        tokens: Character tokens list.
        predicted_bio_tags: Predicted BIO slot tags list.

    Example Input:
        tokens = ['ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'ს']
        predicted_bio_tags = ['B-VERSION', 'B-ROOT', 'I-ROOT', 'I-ROOT', 'B-PFSF', 'I-PFSF', 'B-SUFFIX']

    Example Output:
        "--ა-კეთ-ებ-ს"
    """
    extracted_slots = {}
    current_slot = None
    current_chars = []

    for char, tag in zip(tokens, predicted_bio_tags):
        if tag.startswith("B-"):
            if current_slot:
                extracted_slots[current_slot] = "".join(current_chars)
            current_slot = tag[2:].lower()
            current_chars = [char]
        elif tag.startswith("I-") and current_slot == tag[2:].lower():
            current_chars.append(char)

    if current_slot:
        extracted_slots[current_slot] = "".join(current_chars)

    # Find the last non-empty slot
    active_indices = [i for i, k in enumerate(
        SLOT_KEYS) if extracted_slots.get(k)]
    if not active_indices:
        return ""

    last_active_idx = active_indices[-1]

    # Only include slots up to the last active one
    slot_values = [extracted_slots.get(SLOT_KEYS[i], "")
                   for i in range(last_active_idx + 1)]
    return "-".join(slot_values)


# ==========================================
# 3. MULTI-SEPARATOR BOUNDARY TAGGING (STRATEGY: BOUNDARIES)
# ==========================================

def encode_boundaries(form: str, attr_dict: dict) -> Tuple[List[str], List[int]]:
    """
    Encodes surface characters into boundary delimiter classes (0-3).
    Includes Start-of-Word '^' anchor to assign leading empty preverb/person slots.

    Args:
        form: Raw surface string.
        attr_dict: Form attribute dictionary containing slot values.

    Example Input:
        form = "აკეთებს"
        attr_dict = {"version": "ა", "root": "კეთ", "pfsf": "ებ", "suffix": "ს"}

    Example Output:
        tokens = ['^', 'ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'ს']
        labels = [2, 1, 0, 0, 1, 0, 1, 0]
        # Explanation: 
        # '^' gets 2 for missing preverb and person ('--'),
        # 'ა' (version vowel end) gets 1 ('-'),
        # 'თ' (root end) gets 1 ('-'),
        # 'ბ' (pfsf end) gets 1 ('-').
    """
    chars = ["^"]
    labels = []

    slot_indices = [i for i, k in enumerate(SLOT_KEYS) if attr_dict.get(k)]
    if not slot_indices:
        return ["^"] + list(form), [0] * (len(form) + 1)

    # 1. Assign leading missing slots to '^' anchor (capped at 3)
    leading_nulls = slot_indices[0]
    labels.append(min(leading_nulls, 3) if leading_nulls > 0 else 0)

    # 2. Characters inside active slots
    for idx_in_active, slot_idx in enumerate(slot_indices):
        key = SLOT_KEYS[slot_idx]
        val = attr_dict[key]

        intermediate_nulls = (
            slot_indices[idx_in_active + 1] - slot_idx - 1
            if idx_in_active < len(slot_indices) - 1 else 0
        )

        for c_idx, char in enumerate(val):
            chars.append(char)
            is_slot_end = (c_idx == len(val) - 1)
            is_last_slot = (idx_in_active == len(slot_indices) - 1)

            if is_slot_end and not is_last_slot:
                labels.append(min(1 + intermediate_nulls, 3))
            else:
                labels.append(0)

    return chars, labels


def decode_boundaries(tokens: List[str], predicted_labels: List[int]) -> str:
    """
    Decodes character tokens and boundary class IDs (0-3) into a segmented string.

    Args:
        tokens: Character tokens list (including '^').
        predicted_labels: Boundary class predictions list.

    Example Input:
        tokens = ['^', 'ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'ს']
        predicted_labels = [2, 1, 0, 0, 1, 0, 1, 0]

    Example Output:
        "--ა-კეთ-ებ-ს"
    """
    result = []
    for tok, lbl in zip(tokens, predicted_labels):
        sep = DELIMITER_MAP.get(lbl, "")
        if tok == "^":
            result.append(sep)
        else:
            result.append(tok + sep)
    return "".join(result)


# ==========================================
# 4. PIPELINE ENCODE & DECODE ROUTERS
# ==========================================

def encode(df: pd.DataFrame, strategy: str = "boundaries") -> pd.DataFrame:
    """
    Prepares raw string/list sequence tagging dataset with chosen encoding strategy.

    Args:
        df: Input DataFrame from load.load_and_merge_data().
        strategy: 'boundaries' for multi-separator tagger, or 'bio' for BIO slot tagger.

    Returns:
        pd.DataFrame containing tokens, labels, and raw categorical metadata.

    Example Input (df row):
        form = "აკეთებს"
        form_attr = {"version": "ა", "root": "კეთ", "pfsf": "ებ", "suffix": "ს"}

    Example Output (DataFrame row with strategy='boundaries'):
        tokens: ['^', 'ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'ს']
        labels: [2, 1, 0, 0, 1, 0, 1, 0]
        screeve: "present"
        subject: "3s"
        verb_class: "1"
    """
    records = []
    for _, row in df.iterrows():
        if strategy == "bio":
            tokens, labels = encode_bio(row["form_attr"])
        elif strategy == "boundaries":
            tokens, labels = encode_boundaries(row["form"], row["form_attr"])
        else:
            raise ValueError("strategy must be 'boundaries' or 'bio'")

        records.append({
            "form_id": row["form_id"],
            "word_id": row["word_id"],
            "tokens": tokens,
            "labels": labels,
            "screeve": row["screeve"],
            "subject": row["subject"],
            "verb_class": row["verb_class"]
        })
    return pd.DataFrame(records)


def encode_with_ids(df: pd.DataFrame, vocabs: Dict[str, Dict[str, int]], strategy: str = "boundaries") -> pd.DataFrame:
    """
    Prepares dataset converted to integer IDs ready for PyTorch Tensors.
    Includes metadata integer IDs for screeve, subject, and verb class.

    Args:
        df: Input DataFrame from load.load_and_merge_data().
        vocabs: Vocabularies dictionary built by build_vocabularies().
        strategy: 'boundaries' or 'bio'.

    Returns:
        pd.DataFrame with integer ID lists and scalar integer IDs for metadata.

    Example Output (DataFrame row with strategy='boundaries'):
        token_ids: [2, 3, 10, 12, 15, 12, 18, 22]
        label_ids: [2, 1, 0, 0, 1, 0, 1, 0]
        screeve_id: 1
        subject_id: 4
        class_id: 0
        seq_len: 8
    """
    raw_encoded = encode(df, strategy=strategy)
    records = []

    c2id = vocabs["char2id"]
    l2id = vocabs["label2id"]
    s2id = vocabs["screeve2id"]
    subj2id = vocabs["subject2id"]
    v2id = vocabs["class2id"]

    for _, row in raw_encoded.iterrows():
        token_ids = [c2id.get(tok, c2id["<UNK>"]) for tok in row["tokens"]]
        label_ids = [l2id[lbl] for lbl in row["labels"]]

        records.append({
            "token_ids": token_ids,
            "label_ids": label_ids,
            "screeve_id": s2id.get(row["screeve"], 0),
            "subject_id": subj2id.get(row["subject"], 0),
            "class_id": v2id.get(row["verb_class"], 0),
            "seq_len": len(token_ids)
        })
    return pd.DataFrame(records)


def decode(tokens: List[str], predicted_labels: List[Any], strategy: str = "boundaries") -> str:
    """
    Decodes predictions back to segmented string based on chosen strategy.

    Args:
        tokens: Character tokens list.
        predicted_labels: Predicted boundary IDs (int) or BIO tags (str).
        strategy: 'boundaries' or 'bio'.

    Returns:
        Segmented target string with hyphen slot separators.

    Example Input (strategy='boundaries'):
        tokens = ['^', 'ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'ს']
        predicted_labels = [2, 1, 0, 0, 1, 0, 1, 0]
    Example Output:
        "--ა-კეთ-ებ-ს"

    Example Input (strategy='bio'):
        tokens = ['ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'ს']
        predicted_labels = ['B-VERSION', 'B-ROOT', 'I-ROOT', 'I-ROOT', 'B-PFSF', 'I-PFSF', 'B-SUFFIX']
    Example Output:
        "--ა-კეთ-ებ-ს"
    """
    if strategy == "bio":
        return decode_bio(tokens, predicted_labels)
    elif strategy == "boundaries":
        return decode_boundaries(tokens, predicted_labels)
    else:
        raise ValueError("strategy must be 'boundaries' or 'bio'")
