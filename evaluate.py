"""
evaluate.py
-----------
Comprehensive evaluation suite for Georgian verb morpheme segmentation models.

Metrics Computed:
=================
1. Exact Match Ratio (%):
   - Percentage of verb forms where the predicted segmented string matches gold 100% 
     character-for-character (e.g., "--ა-კეთ-ებ-ს" == "--ა-კეთ-ებ-ს").

2. Normalized Levenshtein Edit Distance:
   - Average edit distance (insertions, deletions, substitutions) per character.
   - Range: 0.0 (perfect) to >1.0. Lower is better.

3. Positional Slot Accuracy (%):
   - Compares predicted slot vs. gold slot across all 6 positional slots (0 to 5).
   - Includes empty-to-empty matches ("" == "").

4. Morpheme Slot Precision, Recall & F1 (%):
   - Strict evaluation over NON-EMPTY morphemes:
     * Precision = Correctly extracted morphemes / Total predicted morphemes
     * Recall    = Correctly extracted morphemes / Total gold morphemes
     * F1        = Harmonic mean of Precision and Recall
"""

import pandas as pd
from typing import List, Dict, Any
from load import SLOT_KEYS


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes standard Levenshtein edit distance between two strings.

    Example:
        s1 = "--ა-კეთ-ებ-ს"
        s2 = "-ა-კეთ-ებ-ს"
        returns 1 (one deletion)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def parse_segmented_slots(segmented_str: str) -> List[str]:
    """
    Parses a segmented string into its 6 template slot values.

    Args:
        segmented_str: Hyphen-delimited string (e.g., "--ა-კეთ-ებ-ს")

    Returns:
        List of 6 strings corresponding to SLOT_KEYS:
        [prefix, person, version, root, pfsf, suffix]

    Example Input:
        "--ა-კეთ-ებ-ს"
    Example Output:
        ["", "", "ა", "კეთ", "ებ", "ს"]
    """
    if not segmented_str:
        return [""] * len(SLOT_KEYS)

    parts = segmented_str.split("-")

    # Pad or truncate to ensure exactly 6 slots
    slots = parts + [""] * (len(SLOT_KEYS) - len(parts))
    return slots[:len(SLOT_KEYS)]


def evaluate_predictions(gold_targets: List[str], pred_targets: List[str]) -> Dict[str, float]:
    """
    Calculates comprehensive evaluation metrics for a set of predictions.

    Args:
        gold_targets: List of ground-truth segmented strings (e.g., ["--ა-კეთ-ებ-ს"])
        pred_targets: List of model-predicted segmented strings

    Returns:
        Dict containing:
          - exact_match: Percentage of 100% identical string matches
          - norm_edit_dist: Normalized character edit distance
          - positional_slot_acc: Percentage of positional slots matching (incl. empty slots)
          - morpheme_precision: Precision on non-empty morpheme extractions
          - morpheme_recall: Recall on non-empty morpheme extractions
          - morpheme_f1: F1 score on non-empty morpheme extractions
    """
    exact_matches = 0
    total_edit_dist = 0.0
    total_chars = 0

    positional_correct = 0
    positional_total = 0

    tp, fp, fn = 0, 0, 0

    for gold, pred in zip(gold_targets, pred_targets):
        # 1. Exact Match
        if gold == pred:
            exact_matches += 1

        # 2. Levenshtein Distance
        dist = levenshtein_distance(gold, pred)
        total_edit_dist += dist
        total_chars += max(len(gold), 1)

        # 3. Slot Parsing & Evaluation
        gold_slots = parse_segmented_slots(gold)
        pred_slots = parse_segmented_slots(pred)

        for slot_idx in range(len(SLOT_KEYS)):
            g_val = gold_slots[slot_idx]
            p_val = pred_slots[slot_idx]

            # Positional Accuracy (counts empty == empty)
            if g_val == p_val:
                positional_correct += 1
            positional_total += 1

            # Strict Morpheme F1 Logic (ignores empty == empty)
            if g_val != "" and p_val != "":
                if g_val == p_val:
                    tp += 1  # Correct non-empty morpheme
                else:
                    fp += 1  # Wrong predicted morpheme
                    fn += 1  # Missed gold morpheme
            elif g_val == "" and p_val != "":
                fp += 1      # Predicted a morpheme where none exists
            elif g_val != "" and p_val == "":
                fn += 1      # Missed an existing morpheme

    # Compute Final Aggregates
    exact_match_pct = (exact_matches / max(len(gold_targets), 1)) * 100
    norm_edit_dist = total_edit_dist / max(total_chars, 1)
    pos_acc_pct = (positional_correct / max(positional_total, 1)) * 100

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    return {
        "exact_match": round(exact_match_pct, 2),
        "norm_edit_dist": round(norm_edit_dist, 4),
        "positional_slot_acc": round(pos_acc_pct, 2),
        "morpheme_precision": round(precision * 100, 2),
        "morpheme_recall": round(recall * 100, 2),
        "morpheme_f1": round(f1 * 100, 2)
    }


def export_error_analysis(
    text_list, gold_list, pred_list, output_path="error_analysis.csv"
):
    """Filters and saves predictions that failed exact match for qualitative review."""
    analysis_data = []
    for text, gold, pred in zip(text_list, gold_list, pred_list):
        if gold != pred:
            analysis_data.append(
                {
                    "Input Form": text,
                    "Gold Segmentation": gold,
                    "Predicted Segmentation": pred,
                }
            )

    df_errors = pd.DataFrame(analysis_data)
    if not df_errors.empty:
        df_errors.to_csv(output_path, index=False)
        print(
            f"🔍 Exported {len(df_errors)} error cases to '{output_path}' for"
            " linguistic debugging."
        )
    else:
        print("🎉 Zero errors found! Perfect match across the test set.")
    return df_errors
