"""
make_mini_dataset.py
--------------------
Generates miniature TSV files from specified verb lemmas for clean local testing.
"""

import argparse
import pandas as pd

# Default standard regular verbs that have full 6-slot morpheme annotations
DEFAULT_LEMMAS = ["დაწერა"]


def create_mini_dataset(
    words_path: str = "words.tsv",
    forms_path: str = "forms.tsv",
    target_lemmas: list = None,
    num_verbs: int = 4,
    out_words_path: str = "words_mini.tsv",
    out_forms_path: str = "forms_mini.tsv"
):
    print(f"Reading {words_path} and {forms_path}...")
    words_df = pd.read_csv(words_path, sep="\t")
    forms_df = pd.read_csv(forms_path, sep="\t")

    # Filter for verbs only
    verb_words = words_df[words_df["pos"] == "verb"].copy()

    if target_lemmas:
        # Filter by explicitly provided lemmas
        selected = verb_words[verb_words["lemma"].isin(target_lemmas)]
        if selected.empty:
            print(
                f"Warning: None of {target_lemmas} found in {words_path}. Falling back to first {num_verbs} verbs.")
            verb_word_ids = verb_words["word_id"].unique()[:num_verbs]
        else:
            verb_word_ids = selected["word_id"].unique()
    else:
        # Try matching default regular lemmas first
        selected = verb_words[verb_words["lemma"].isin(DEFAULT_LEMMAS)]
        if not selected.empty:
            verb_word_ids = selected["word_id"].unique()
        else:
            # Fallback if default lemmas are absent from your file
            verb_word_ids = verb_words["word_id"].unique()[:num_verbs]

    mini_words = words_df[words_df["word_id"].isin(verb_word_ids)]
    mini_forms = forms_df[forms_df["word_id"].isin(verb_word_ids)]

    mini_words.to_csv(out_words_path, sep="\t", index=False)
    mini_forms.to_csv(out_forms_path, sep="\t", index=False)

    chosen_lemmas = mini_words["lemma"].tolist()
    print(f"\nDone! Created mini dataset with lemmas: {chosen_lemmas}")
    print(f"  - {out_words_path}: {len(mini_words)} verb entries")
    print(f"  - {out_forms_path}: {len(mini_forms)} form entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words_path", type=str, default="words.tsv")
    parser.add_argument("--forms_path", type=str, default="forms.tsv")
    parser.add_argument("--lemmas", nargs="+", default=None,
                        help="Specific lemmas to include (e.g. --lemmas გაკეთება დაწერა)")
    parser.add_argument("--num_verbs", type=int, default=4)
    args = parser.parse_args()

    create_mini_dataset(
        words_path=args.words_path,
        forms_path=args.forms_path,
        target_lemmas=args.lemmas,
        num_verbs=args.num_verbs
    )
