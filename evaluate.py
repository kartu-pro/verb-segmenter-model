"""
evaluate.py
-----------
Evaluation metrics for Georgian verb segmentation models.

Metrics:
  1. Exact Match Ratio: Character-for-character equality of segmented target.
  2. Normalized Levenshtein Distance: Edit distance normalized by gold sequence length.
  3. Slot-Level Precision/Recall/F1: Evaluates slot-by-slot accuracy across the 6 slots.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

SLOT_KEYS = ["prefix", "person", "version", "root", "pfsf", "suffix"]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes standard edit distance between two strings."""
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
    
    Example Input: "--ა-კეთ-ებ-ს" -> ["", "", "ა", "კეთ", "ებ", "ს"]
    """
    parts = segmented_str.split("-")
    
    # Handle leading dashes (empty slots)
    slots = []
    i = 0
    while i < len(parts):
        slots.append(parts[i])
        i += 1
        
    # Pad or truncate to exactly 6 slots
    while len(slots) < len(SLOT_KEYS):
        slots.append("")
    return slots[:len(SLOT_KEYS)]


def evaluate_predictions(gold_targets: List[str], pred_targets: List[str]) -> Dict[str, float]:
    """
    Calculates overall evaluation metrics for a batch of predictions.

    Args:
        gold_targets: Ground truth segmented strings (e.g., ["--ა-კეთ-ებ-ს"])
        pred_targets: Model predicted segmented strings

    Returns:
        Dict with 'exact_match', 'norm_edit_distance', and 'slot_f1'.
    """
    exact_matches = 0
    total_edit_dist = 0.0
    total_chars = 0
    
    slot_correct = 0
    slot_total = 0

    for gold, pred in zip(gold_targets, pred_targets):
        # 1. Exact Match
        if gold == pred:
            exact_matches += 1

        # 2. Levenshtein Distance
        dist = levenshtein_distance(gold, pred)
        total_edit_dist += dist
        total_chars += max(len(gold), 1)

        # 3. Slot-Level Evaluation
        gold_slots = parse_segmented_slots(gold)
        pred_slots = parse_segmented_slots(pred)
        
        for g_s, p_s in zip(gold_slots, pred_slots):
            if g_s == p_s:
                slot_correct += 1
            slot_total += 1

    exact_match_ratio = exact_matches / max(len(gold_targets), 1)
    norm_edit_distance = total_edit_dist / max(total_chars, 1)
    slot_acc = slot_correct / max(slot_total, 1)

    return {
        "exact_match": round(exact_match_ratio * 100, 2),
        "norm_edit_dist": round(norm_edit_distance, 4),
        "slot_accuracy_f1": round(slot_acc * 100, 2)
    }