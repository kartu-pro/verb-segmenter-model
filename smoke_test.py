"""
test_local.py
-------------
Comprehensive local verification and smoke test script. 
1. Runs a syntax/compile check across all core python modules.
2. Tests data parsing, encoding strategies, decoders, and evaluation metrics.
3. Tests training argument compatibility without requiring heavy GPU runs.
4. Optionally runs a live training smoke test with google/byt5-small IF running in Google Colab.
"""

import py_compile
import sys
import load
import seq_2_seq
import seq_tagger
import evaluate
import train_seq2seq


def is_collab_environment():
    """Detects if code is executing inside a Google Colab notebook."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def run_smoke_test():
    print("=" * 60)
    print("RUNNING LOCAL SMOKE TEST & VERIFICATION SUITE")
    print("=" * 60)

    # 0. Test Syntax & Compilation (Catches typos/syntax errors early)
    print("\n[0/7] Running Python Compilation Check (py_compile)...")
    modules_to_check = [
        "load.py",
        "seq_2_seq.py",
        "seq_tagger.py",
        "evaluate.py",
        "train_seqtagger.py",
        "train_seq2seq.py",
        "experiments.py"
    ]

    compilation_failed = False
    for mod in modules_to_check:
        try:
            py_compile.compile(mod, doraise=True)
            print(f"  ✓ Compiled successfully: {mod}")
        except Exception as e:
            print(f"  ❌ Compilation error in {mod}: {e}")
            compilation_failed = True

    if compilation_failed:
        print("\n❌ STOPPING: Fix compilation/syntax errors before proceeding.")
        sys.exit(1)

    # 1. Test Data Loading
    print("\n[1/7] Testing load.py...")
    df = load.load_and_merge_data("words_mini.tsv", "forms_mini.tsv")
    print(
        f"  ✓ Loaded {len(df)} forms across {df['word_id'].nunique()} verb lemmas.")

    # 2. Test Seq2Seq Encoding & Decoding
    print("\n[2/7] Testing seq_2_seq.py...")
    s2s_single = seq_2_seq.encode(df, mode="single")
    sample_single_input = s2s_single.iloc[0]["input_text"]
    sample_single_target = s2s_single.iloc[0]["target_text"]
    decoded_single = seq_2_seq.decode(sample_single_target, mode="single")
    print(f"  ✓ Single Prompt: {sample_single_input}")
    print(f"  ✓ Single Target: {sample_single_target}")
    print(f"  ✓ Decoded:       {decoded_single[0]}")

    s2s_group6 = seq_2_seq.encode(df, mode="group6")
    sample_g6_input = s2s_group6.iloc[0]["input_text"]
    sample_g6_target = s2s_group6.iloc[0]["target_text"]
    decoded_g6 = seq_2_seq.decode(sample_g6_target, mode="group6")
    print(f"  ✓ Group6 Prompt: {sample_g6_input[:70]}...")
    print(f"  ✓ Group6 Target Count: {len(decoded_g6)} forms")

    # 3. Test Sequence Tagger (Boundaries Strategy)
    print("\n[3/7] Testing seq_tagger.py (Boundaries Strategy)...")
    bound_df = seq_tagger.encode(df, strategy="boundaries")
    vocabs_bound = seq_tagger.build_vocabularies(df, strategy="boundaries")
    bound_ids_df = seq_tagger.encode_with_ids(
        df, vocabs_bound, strategy="boundaries")

    sample_toks = bound_df.iloc[0]["tokens"]
    sample_lbls = bound_df.iloc[0]["labels"]
    decoded_bound = seq_tagger.decode(
        sample_toks, sample_lbls, strategy="boundaries")

    print(f"  ✓ Tokens:        {sample_toks}")
    print(f"  ✓ Boundary IDs:  {sample_lbls}")
    print(f"  ✓ Decoded:       {decoded_bound}")
    print(f"  ✓ Token IDs:     {bound_ids_df.iloc[0]['token_ids']}")

    # 4. Test Sequence Tagger (BIO Strategy)
    print("\n[4/7] Testing seq_tagger.py (BIO Strategy)...")
    bio_df = seq_tagger.encode(df, strategy="bio")
    vocabs_bio = seq_tagger.build_vocabularies(df, strategy="bio")
    bio_ids_df = seq_tagger.encode_with_ids(df, vocabs_bio, strategy="bio")

    sample_bio_toks = bio_df.iloc[0]["tokens"]
    sample_bio_lbls = bio_df.iloc[0]["labels"]
    decoded_bio = seq_tagger.decode(
        sample_bio_toks, sample_bio_lbls, strategy="bio")

    print(f"  ✓ Tokens:        {sample_bio_toks}")
    print(f"  ✓ BIO Tags:      {sample_bio_lbls}")
    print(f"  ✓ Decoded:       {decoded_bio}")

    # 5. Test Evaluation Metrics
    print("\n[5/7] Testing evaluate.py...")
    gold_targets = [
        "-".join([df.iloc[i]["form_attr"].get(k, "") for k in load.SLOT_KEYS])
        for i in range(len(df))
    ]

    # Gold vs Gold (Expect 100% exact match)
    perfect_metrics = evaluate.evaluate_predictions(gold_targets, gold_targets)
    print(f"  ✓ Perfect Match Metrics: {perfect_metrics}")

    # Gold vs Realistic Corrupted targets (Testing partial Levenshtein and slot F1)
    corrupted_targets = [g.replace("წერ", "ცერ") for g in gold_targets]
    imperfect_metrics = evaluate.evaluate_predictions(
        gold_targets, corrupted_targets)
    print(f"  ✓ Imperfect Match Metrics: {imperfect_metrics}")

    # 6. Test Training Arguments Initialization
    print("\n[6/7] Testing Training Arguments Compatibility...")
    try:
        from transformers import Seq2SeqTrainingArguments
        args = Seq2SeqTrainingArguments(
            output_dir="./test_output",
            eval_strategy="epoch"
        )
        print("  ✓ Seq2SeqTrainingArguments initialized successfully with 'eval_strategy'.")
    except ImportError:
        print("  ℹ️ 'transformers' not installed locally. Skipping training args test.")
    except Exception as e:
        print(f"  ❌ Seq2SeqTrainingArguments initialization failed: {e}")
        sys.exit(1)

    # 7. Conditional Live Training Smoke Test (Colab Only)
    print("\n[7/7] Checking Live Training Smoke Test Environment...")
    if is_collab_environment():
        print("  🚀 Google Colab environment detected! Running google/byt5-small training smoke test...")
        try:
            train_seq2seq.run_training(
                words_path="words.tsv",
                forms_path="forms.tsv",
                mode="single",
                epochs=1,
                max_samples=300,
                model_name="google/byt5-small",
                smoke_test=True
            )
            print("  ✓ Live training smoke test completed successfully!")
        except Exception as e:
            print(f"  ❌ Live training smoke test failed: {e}")
            sys.exit(1)
    else:
        print("  ℹ️ Not in a Colab environment (local run). Skipping live model training smoke test to save time/space.")

    print("\n" + "=" * 60)
    print("SUCCESS: ALL LOCAL VERIFICATIONS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
