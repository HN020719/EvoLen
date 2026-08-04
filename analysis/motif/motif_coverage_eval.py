#!/usr/bin/env python3
import argparse
import csv
import itertools
import re
import statistics
from datetime import datetime
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_TOKENIZER_PATHS = {
    "baseline_bpe_2048": "tokenizer_evaluation/baseline_bpe/vocab_2048/2048_tokenizer.json",
    "baseline_bpe_3072": "tokenizer_evaluation/baseline_bpe/vocab_3072/3072_tokenizer.json",
    "baseline_bpe_4096": "tokenizer_evaluation/baseline_bpe/vocab_4096/4096_tokenizer.json",
    "baseline_bpe_5120": "tokenizer_evaluation/baseline_bpe/vocab_5120/5120_tokenizer.json",
    "merge_uni_len2_2048": "tokenizer_evaluation/merge_bpe/vocab_2048/merge_tokenizer_unigram_len2.json",
    "merge_uni_len2_3072": "tokenizer_evaluation/merge_bpe/vocab_3072/merge_tokenizer_unigram_len2.json",
    "merge_uni_len2_4096": "tokenizer_evaluation/merge_bpe/vocab_4096/merge_tokenizer_unigram_len2.json",
    "merge_uni_len2_5120": "tokenizer_evaluation/merge_bpe/vocab_5120/merge_tokenizer_unigram_len2.json",
}

# Custom (non-HuggingFace-`tokenizers`) tokenizers.
# Path is a directory containing the required assets.
CUSTOM_TOKENIZER_PATHS = {
    "DNAMotifTokenizer": "tokenizer_files/DNAMotifTokenizer_official",
}

ALLOWED_MOTIF_CHARS = re.compile(r"[^ACGTNRYWSMKBDHV*]")


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate motif coverage/fragmentation for DNA tokenizers."
    )
    parser.add_argument("--motif-file", required=True, help="Path to motif txt file")
    parser.add_argument(
        "--output-dir",
        default="tokenizer_evaluation/motif_eval_outputs",
        help="Output dir for csv files",
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=5,
        help="Skip motifs with more than this many '*'",
    )
    parser.add_argument(
        "--min-motif-len",
        type=int,
        default=1,
        help="Ignore motifs shorter than this length",
    )
    parser.add_argument(
        "--test-seq",
        default="TCCTGCCTCAGCCAAAA",
        help="Sanity check sequence for [UNK]",
    )
    return parser.parse_args()


def load_raw_motifs(path: Path, min_motif_len: int) -> list[str]:
    motifs = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            motif_field = line.split()[0]
            motif = ALLOWED_MOTIF_CHARS.sub("", motif_field.upper())
            if len(motif) >= min_motif_len:
                motifs.add(motif)
    return sorted(motifs)


def expand_motif(motif: str, max_stars: int) -> list[str]:
    if "*" not in motif:
        return [motif]

    star_count = motif.count("*")
    if star_count > max_stars:
        return []

    segments = motif.split("*")
    bases = ["A", "C", "G", "T"]
    expanded = []
    for combo in itertools.product(bases, repeat=star_count):
        parts = []
        for i in range(star_count):
            parts.append(segments[i])
            parts.append(combo[i])
        parts.append(segments[-1])
        expanded.append("".join(parts))
    return expanded


def load_tokenizers(root: Path) -> dict:
    tokenizers = {}
    log("Loading tokenizers...")
    for name, rel_path in DEFAULT_TOKENIZER_PATHS.items():
        full_path = (root / rel_path).resolve()
        if not full_path.exists():
            log(f"[Missing] {name}: {full_path}")
            continue
        try:
            tokenizers[name] = Tokenizer.from_file(str(full_path))
            log(f"[OK] {name}")
        except Exception as e:
            log(f"[Failed] {name}: {e}")

    # Custom tokenizers (duck-typed: must expose .encode() returning obj with .tokens, and .get_vocab())
    import sys
    from dna_motif_tokenizer import DNAMotifTokenizer, Trie, TrieNode
    sys.modules["__main__"].Trie = Trie
    sys.modules["__main__"].TrieNode = TrieNode
    for name, rel_path in CUSTOM_TOKENIZER_PATHS.items():
        full_path = (root / rel_path).resolve()
        if not full_path.exists():
            log(f"[Missing] {name}: {full_path}")
            continue
        try:
            tokenizers[name] = DNAMotifTokenizer(str(full_path), seed=42)
            log(f"[OK] {name} (custom)")
        except Exception as e:
            log(f"[Failed] {name}: {e}")
    return tokenizers


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(motifs_raw: list[str], tokenizers: dict[str, Tokenizer], max_stars: int) -> tuple[list[dict], list[dict], list[dict]]:
    full_test_set = set()
    motif_source_map = {}

    skipped_by_star_limit = 0
    log(f"Expanding {len(motifs_raw)} raw motifs...")
    for motif in motifs_raw:
        variants = expand_motif(motif, max_stars)
        if not variants and "*" in motif and motif.count("*") > max_stars:
            skipped_by_star_limit += 1
            continue
        for v in variants:
            full_test_set.add(v)
            motif_source_map[v] = motif

    log(f"Generated {len(full_test_set)} total motif variants")
    log(f"Skipped motifs due to star limit: {skipped_by_star_limit}")

    detail_rows = []
    consistency_buckets = {}  # (tokenizer, original_motif) -> list[token_count]

    log("Running benchmark...")
    variants = sorted(full_test_set)
    for name, tok in tokenizers.items():
        log(f"Evaluating tokenizer: {name} (variants={len(variants)})")
        vocab_set = set(tok.get_vocab().keys())

        for variant in variants:
            motif = motif_source_map[variant]
            motif_len = len(motif)

            encoded = tok.encode(variant, add_special_tokens=False)
            tokens = encoded.tokens
            token_count = len(tokens)

            if token_count > 0 and motif_len > 0:
                avg_token_fraction = sum(len(t) for t in tokens) / float(token_count * motif_len)
            else:
                avg_token_fraction = 0.0

            row = {
                "Tokenizer": name,
                "Original_Motif": motif,
                "Variant": variant,
                "Motif_Length": motif_len,
                "Token_Count": token_count,
                "Is_Perfect": 1 if token_count == 1 else 0,
                "Is_Exact_In_Vocab": 1 if variant in vocab_set else 0,
                "Avg_Token_Fraction": avg_token_fraction,
            }
            detail_rows.append(row)

            key = (name, motif)
            if key not in consistency_buckets:
                consistency_buckets[key] = []
            consistency_buckets[key].append(token_count)

        log(f"Done tokenizer: {name}")

    by_tokenizer = {}
    for r in detail_rows:
        k = r["Tokenizer"]
        if k not in by_tokenizer:
            by_tokenizer[k] = {
                "token_counts": [],
                "is_perfect": [],
                "is_exact": [],
                "fractions": [],
            }
        by_tokenizer[k]["token_counts"].append(r["Token_Count"])
        by_tokenizer[k]["is_perfect"].append(r["Is_Perfect"])
        by_tokenizer[k]["is_exact"].append(r["Is_Exact_In_Vocab"])
        by_tokenizer[k]["fractions"].append(r["Avg_Token_Fraction"])

    summary_rows = []
    for name, vals in by_tokenizer.items():
        tc = vals["token_counts"]
        pf = vals["is_perfect"]
        ex = vals["is_exact"]
        fr = vals["fractions"]

        summary_rows.append(
            {
                "Tokenizer": name,
                "Avg_Tokens_Per_Motif": statistics.mean(tc) if tc else 0.0,
                "Median_Tokens_Per_Motif": statistics.median(tc) if tc else 0.0,
                "Perfect_Match_Rate": (100.0 * sum(pf) / len(pf)) if pf else 0.0,
                "Exact_Vocab_Coverage_Rate": (100.0 * sum(ex) / len(ex)) if ex else 0.0,
                "Avg_Token_Fraction": statistics.mean(fr) if fr else 0.0,
                "Median_Token_Fraction": statistics.median(fr) if fr else 0.0,
            }
        )

    summary_rows.sort(key=lambda x: (x["Avg_Tokens_Per_Motif"], -x["Exact_Vocab_Coverage_Rate"]))

    consistency_rows = []
    for (tok_name, motif), counts in consistency_buckets.items():
        std_val = statistics.stdev(counts) if len(counts) > 1 else 0.0
        consistency_rows.append(
            {
                "Tokenizer": tok_name,
                "Original_Motif": motif,
                "Token_Count_Std": std_val,
            }
        )

    return detail_rows, summary_rows, consistency_rows


def save_outputs(detail_rows: list[dict], summary_rows: list[dict], consistency_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / "motif_variant_results.csv"
    summary_csv = out_dir / "summary_by_tokenizer.csv"
    consistency_csv = out_dir / "consistency_by_motif.csv"

    write_csv(
        detail_csv,
        detail_rows,
        [
            "Tokenizer",
            "Original_Motif",
            "Variant",
            "Motif_Length",
            "Token_Count",
            "Is_Perfect",
            "Is_Exact_In_Vocab",
            "Avg_Token_Fraction",
        ],
    )

    write_csv(
        summary_csv,
        summary_rows,
        [
            "Tokenizer",
            "Avg_Tokens_Per_Motif",
            "Median_Tokens_Per_Motif",
            "Perfect_Match_Rate",
            "Exact_Vocab_Coverage_Rate",
            "Avg_Token_Fraction",
            "Median_Token_Fraction",
        ],
    )

    write_csv(
        consistency_csv,
        consistency_rows,
        ["Tokenizer", "Original_Motif", "Token_Count_Std"],
    )

    log(f"Saved outputs to: {out_dir}")
    log(f"- {detail_csv}")
    log(f"- {summary_csv}")
    log(f"- {consistency_csv}")


def run_unk_check(tokenizers: dict[str, Tokenizer], test_seq: str) -> None:
    log("[UNK] sanity check")
    log(f"Input sequence: {test_seq}")
    for name, tok in tokenizers.items():
        out = tok.encode(test_seq)
        has_unk = "[UNK]" in out.tokens
        status = "FAIL" if has_unk else "OK"
        log(f"- {name}: {status}; tokens={out.tokens}")


def print_summary(summary_rows: list[dict]) -> None:
    log("--- Summary Statistics ---")
    for r in summary_rows:
        log(
            f"{r['Tokenizer']}: avg_tok={r['Avg_Tokens_Per_Motif']:.4f}, "
            f"perfect={r['Perfect_Match_Rate']:.2f}%, "
            f"exact={r['Exact_Vocab_Coverage_Rate']:.2f}%, "
            f"avg_frac={r['Avg_Token_Fraction']:.4f}"
        )


def main() -> None:
    args = parse_args()
    log("Starting motif coverage evaluation")

    root = Path(__file__).resolve().parents[1]
    motif_file = Path(args.motif_file).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    log(f"Motif file: {motif_file}")
    log(f"Output dir: {out_dir}")

    if not motif_file.exists():
        raise FileNotFoundError(f"Motif file not found: {motif_file}")

    motifs_raw = load_raw_motifs(motif_file, min_motif_len=args.min_motif_len)
    log(f"Loaded {len(motifs_raw)} cleaned unique motifs")

    tokenizers = load_tokenizers(root)
    if not tokenizers:
        raise RuntimeError("No tokenizer could be loaded. Check file paths.")
    log(f"Total loaded tokenizers: {len(tokenizers)}")

    detail_rows, summary_rows, consistency_rows = evaluate(
        motifs_raw=motifs_raw,
        tokenizers=tokenizers,
        max_stars=args.max_stars,
    )

    print_summary(summary_rows)
    save_outputs(detail_rows, summary_rows, consistency_rows, out_dir)
    run_unk_check(tokenizers, args.test_seq)
    log("Evaluation completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        raise
