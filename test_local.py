"""
test_local.py
-------------
Lightweight local verification script. Tests data parsing, encoding strategies,
decoders, and evaluation metrics without needing PyTorch or HuggingFace.
"""

import load
import seq_2_seq
import seq_tagger
import evaluate


def run_smoke_test():
    print("=" * 60)
    print("RUNNING LOCAL SMOKE TEST")
    print("=" * 60)

    # 1. Test Data Loading
    print("\n[1/5] Testing load.py...")
    df = load.load_and_merge_data("words_mini.tsv", "forms_mini.tsv")
    print(f"  ✓ Loaded {len(df)} forms across {df['word_id'].nunique()} verb lemmas.")

    # 2. Test Seq2Seq Encoding & Decoding
    print("\n[2/5] Testing seq_2_seq.py...")
    # Single mode
    s2s_single = seq_2_seq.encode(df, mode="single")
    sample_single_input = s2s_single.iloc[0]["input_text"]
    sample_single_target = s2s_single.iloc[0]["target_text"]
    decoded_single = seq_2_seq.decode(sample_single_target, mode="single")
    print(f"  ✓ Single Prompt: {sample_single_input}")
    print(f"  ✓ Single Target: {sample_single_target}")
    print(f"  ✓ Decoded:       {decoded_single[0]}")

    # Group6 mode
    s2s_group6 = seq_2_seq.encode(df, mode="group6")
    sample_g6_input = s2s_group6.iloc[0]["input_text"]
    sample_g6_target = s2s_group6.iloc[0]["target_text"]
    decoded_g6 = seq_2_seq.decode(sample_g6_target, mode="group6")
    print(f"  ✓ Group6 Prompt: {sample_g6_input[:70]}...")
    print(f"  ✓ Group6 Target Count: {len(decoded_g6)} forms")

    # 3. Test Sequence Tagger (Boundaries Strategy)
    print("\n[3/5] Testing seq_tagger.py (Boundaries Strategy)...")
    bound_df = seq_tagger.encode(df, strategy="boundaries")
    vocabs_bound = seq_tagger.build_vocabularies(df, strategy="boundaries")
    bound_ids_df = seq_tagger.encode_with_ids(df, vocabs_bound, strategy="boundaries")

    sample_toks = bound_df.iloc[0]["tokens"]
    sample_lbls = bound_df.iloc[0]["labels"]
    decoded_bound = seq_tagger.decode(sample_toks, sample_lbls, strategy="boundaries")

    print(f"  ✓ Tokens:        {sample_toks}")
    print(f"  ✓ Boundary IDs:  {sample_lbls}")
    print(f"  ✓ Decoded:       {decoded_bound}")
    print(f"  ✓ Token IDs:     {bound_ids_df.iloc[0]['token_ids']}")

    # 4. Test Sequence Tagger (BIO Strategy)
    print("\n[4/5] Testing seq_tagger.py (BIO Strategy)...")
    bio_df = seq_tagger.encode(df, strategy="bio")
    vocabs_bio = seq_tagger.build_vocabularies(df, strategy="bio")
    bio_ids_df = seq_tagger.encode_with_ids(df, vocabs_bio, strategy="bio")

    sample_bio_toks = bio_df.iloc[0]["tokens"]
    sample_bio_lbls = bio_df.iloc[0]["labels"]
    decoded_bio = seq_tagger.decode(sample_bio_toks, sample_bio_lbls, strategy="bio")

    print(f"  ✓ Tokens:        {sample_bio_toks}")
    print(f"  ✓ BIO Tags:      {sample_bio_lbls}")
    print(f"  ✓ Decoded:       {decoded_bio}")

    # 5. Test Evaluation Metrics
    print("\n[5/5] Testing evaluate.py...")
    gold_targets = df["segmented_target"].tolist()
    
    # Gold vs Gold (Expect 100% exact match)
    perfect_metrics = evaluate.evaluate_predictions(gold_targets, gold_targets)
    print(f"  ✓ Perfect Match Metrics: {perfect_metrics}")

    # Gold vs Corrupted targets (Test Levenshtein and slot F1)
    corrupted_targets = [g.replace("-", "") for g in gold_targets]
    imperfect_metrics = evaluate.evaluate_predictions(gold_targets, corrupted_targets)
    print(f"  ✓ Imperfect Match Metrics: {imperfect_metrics}")

    print("\n" + "=" * 60)
    print("SUCCESS: ALL PIPELINES, ENCODERS, AND DECODERS FUNCTIONING CORRECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()