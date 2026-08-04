"""
Ablation study: build tokenizer variants that each remove one design choice from Len2.

Variants:
  1. no_partition:  Reuse existing baseline BPE vocab + length² scoring (removes phyloP stratification)
  2. no_priority:   Train 3 category BPEs but merge with random/uniform order + length² scoring (removes conserved-first priority)
  3. no_length:     Train 3 category BPEs with priority merge but uniform scoring (removes length² scoring)

The full Len2 and baseline BPE already exist; this script produces the three ablation variants.
"""

import argparse
import json
import os
import random

from tokenizers import Tokenizer
from tokenizers.models import Unigram


def load_vocab(path):
    """Load vocab dict from a tokenizer JSON, handling both BPE and Unigram formats."""
    try:
        return Tokenizer.from_file(path).get_vocab()
    except Exception:
        d = json.load(open(path))
        vocab = d["model"]["vocab"]
        if isinstance(vocab, dict):
            return vocab
        elif isinstance(vocab, list):
            return {t: i for i, (t, _) in enumerate(vocab)}
        raise ValueError(f"Unknown vocab format in {path}")


def ensure_essentials(vocab_list, vocab_size):
    """Ensure essential tokens are present and truncate to vocab_size."""
    essentials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "A", "T", "C", "G", "N"]
    for token in essentials:
        if token not in vocab_list:
            vocab_list.append(token)
    return vocab_list[:vocab_size]


def build_unigram_with_scoring(vocab_list, scoring_fn, output_path):
    """Build a Unigram tokenizer from vocab_list with a given scoring function."""
    vocab_entries = []
    for token in vocab_list:
        score = scoring_fn(token)
        vocab_entries.append((token, score))

    # Find [UNK] index so Unigram knows which token to use for unknowns
    unk_id = next((i for i, (t, _) in enumerate(vocab_entries) if t == "[UNK]"), None)
    if unk_id is None:
        raise ValueError(f"[UNK] not found in vocab — ensure_essentials should have added it")

    tokenizer = Tokenizer(Unigram(vocab_entries, unk_id))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Saved: {output_path} ({len(vocab_entries)} tokens, unk_id={unk_id})")


def len2_scoring(token):
    """Length-squared scoring: score(t) = |t|^2."""
    return len(token) ** 2


def length_scoring(token):
    """Length scoring: score(t) = |t| (removes the squared emphasis)."""
    return len(token)


# ─── Priority-based merge (original Len2 logic) ────────────────────────────────

def priority_merge(v_con, v_neu, v_acc, vocab_size):
    """Merge vocabularies with conserved-first priority (original Len2)."""
    vocab_imp = set(v_con.keys())
    vocab_med = set(v_neu.keys())
    vocab_non = set(v_acc.keys())

    all_three = vocab_imp & vocab_med & vocab_non
    unique_imp = vocab_imp - vocab_med - vocab_non
    imp_med_only = (vocab_imp & vocab_med) - vocab_non
    unique_med = vocab_med - vocab_imp - vocab_non

    final = []
    final.extend(list(all_three))
    final.extend(list(unique_imp))
    final.extend(list(imp_med_only))

    remaining = vocab_size - len(final)
    if remaining < 0:
        final = final[:vocab_size]
    else:
        unique_med_tokens = sorted(unique_med, key=lambda t: v_neu[t])
        final.extend(unique_med_tokens[:remaining])

    return ensure_essentials(final, vocab_size)


def random_merge(v_con, v_neu, v_acc, vocab_size):
    """Merge vocabularies with random order (no priority)."""
    all_tokens = set(v_con.keys()) | set(v_neu.keys()) | set(v_acc.keys())
    token_list = list(all_tokens)
    random.seed(42)
    random.shuffle(token_list)
    return ensure_essentials(token_list, vocab_size)


# ─── Ablation variant builders ──────────────────────────────────────────────────

def build_no_partition(baseline_bpe_path, vocab_size, out_dir):
    """
    Ablation: remove phyloP partitioning.
    Reuse existing baseline BPE vocab + length² scoring.
    """
    print("\n=== Ablation: no_partition (baseline BPE vocab + len² scoring) ===")
    tok = Tokenizer.from_file(baseline_bpe_path)
    vocab = tok.get_vocab()
    vocab_list = sorted(vocab.keys(), key=lambda t: vocab[t])
    vocab_list = ensure_essentials(vocab_list, vocab_size)

    output_path = os.path.join(out_dir, "ablation_no_partition.json")
    build_unigram_with_scoring(vocab_list, len2_scoring, output_path)


def build_no_priority(con_path, neu_path, acc_path, vocab_size, out_dir):
    """
    Ablation: remove conserved-first priority.
    Use 3 category BPEs but merge with random order, then apply length² scoring.
    """
    print("\n=== Ablation: no_priority (random merge + len² scoring) ===")
    v_con = load_vocab(con_path)
    v_neu = load_vocab(neu_path)
    v_acc = load_vocab(acc_path)

    vocab_list = random_merge(v_con, v_neu, v_acc, vocab_size)

    output_path = os.path.join(out_dir, "ablation_no_priority.json")
    build_unigram_with_scoring(vocab_list, len2_scoring, output_path)


def build_no_length(con_path, neu_path, acc_path, vocab_size, out_dir):
    """
    Ablation: remove length-aware scoring.
    Use 3 category BPEs with priority merge, but uniform token scores.
    """
    print("\n=== Ablation: no_length (priority merge + uniform scoring) ===")
    v_con = load_vocab(con_path)
    v_neu = load_vocab(neu_path)
    v_acc = load_vocab(acc_path)

    vocab_list = priority_merge(v_con, v_neu, v_acc, vocab_size)

    output_path = os.path.join(out_dir, "ablation_no_length.json")
    build_unigram_with_scoring(vocab_list, length_scoring, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Build ablation tokenizer variants for Len2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-bpe", required=True,
        help="Path to existing baseline BPE tokenizer.json (whole-genome, no phyloP)",
    )
    parser.add_argument(
        "--category-bpe-dir", required=True,
        help="Directory with existing con/neu/acc_tokenizer.json (from train_len2_bpe_multi.py)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=5120,
        help="Fixed vocabulary size (default: 5120)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Base output directory for ablation tokenizers",
    )
    parser.add_argument(
        "--variants", nargs="+",
        default=["no_partition", "no_priority", "no_length"],
        choices=["no_partition", "no_priority", "no_length"],
        help="Which ablation variants to build",
    )

    args = parser.parse_args()
    vocab_size = args.vocab_size

    out_dir = os.path.join(args.output_dir, f"vocab_{vocab_size}")

    # Paths to existing category-specific BPEs (from train_len2_bpe_multi.py)
    cat_dir = os.path.join(args.category_bpe_dir, f"vocab_{vocab_size}")
    con_path = os.path.join(cat_dir, "con_tokenizer.json")
    neu_path = os.path.join(cat_dir, "neu_tokenizer.json")
    acc_path = os.path.join(cat_dir, "acc_tokenizer.json")

    if "no_partition" in args.variants:
        build_no_partition(args.baseline_bpe, vocab_size, out_dir)

    if "no_priority" in args.variants:
        if not all(os.path.exists(p) for p in [con_path, neu_path, acc_path]):
            print(f"Skipping no_priority: missing category BPEs in {cat_dir}")
        else:
            build_no_priority(con_path, neu_path, acc_path, vocab_size, out_dir)

    if "no_length" in args.variants:
        if not all(os.path.exists(p) for p in [con_path, neu_path, acc_path]):
            print(f"Skipping no_length: missing category BPEs in {cat_dir}")
        else:
            build_no_length(con_path, neu_path, acc_path, vocab_size, out_dir)


if __name__ == "__main__":
    main()
