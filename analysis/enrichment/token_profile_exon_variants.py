#!/usr/bin/env python3
import argparse
import csv
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pysam
from tokenizers import Tokenizer


BedInterval = Tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exon-only token profiling across tokenizer variants.")
    p.add_argument("--fasta", required=True, help="Genome FASTA path")
    p.add_argument("--bed", required=True, help="Exon BED path")
    p.add_argument(
        "--out-dir",
        default="token_profile_exon_all4",
        help="Output directory",
    )
    p.add_argument(
        "--vocabs",
        default="2048,3072,4096,5120",
        help="Comma-separated vocab sizes",
    )
    p.add_argument(
        "--max-regions",
        type=int,
        default=100000,
        help="Reservoir sample max intervals",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-n", action="store_true", help="Keep sequences with non-ACGT")
    return p.parse_args()


def is_canonical(chrom: str) -> bool:
    if chrom in {"chrX", "chrY"}:
        return True
    if not chrom.startswith("chr"):
        return False
    c = chrom[3:]
    return c.isdigit() and 1 <= int(c) <= 22


def reservoir_sample_bed(path: str, max_regions: int, rng: random.Random) -> List[BedInterval]:
    sample: List[BedInterval] = []
    seen = 0
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            chrom = cols[0]
            if not is_canonical(chrom):
                continue
            try:
                s = int(cols[1]); e = int(cols[2])
            except ValueError:
                continue
            if s < 0 or e <= s:
                continue
            seen += 1
            iv = (chrom, s, e)
            if len(sample) < max_regions:
                sample.append(iv)
            else:
                j = rng.randint(1, seen)
                if j <= max_regions:
                    sample[j - 1] = iv
    return sample


def gc_frac(tok: str) -> float:
    if not tok:
        return 0.0
    tok = tok.upper()
    return sum(1 for c in tok if c in {"G", "C"}) / len(tok)


def tokenize_intervals(
    fasta: pysam.FastaFile,
    intervals: Sequence[BedInterval],
    tok: Tokenizer,
    keep_n: bool,
) -> Tuple[Counter, List[int], List[float], int]:
    counts = Counter()
    lengths: List[int] = []
    gcs: List[float] = []
    used_regions = 0

    for chrom, s, e in intervals:
        seq = fasta.fetch(chrom, s, e).upper()
        if not seq:
            continue
        if (not keep_n) and any(ch not in {"A", "C", "G", "T"} for ch in seq):
            continue
        enc = tok.encode(seq)
        if not enc.tokens:
            continue
        used_regions += 1
        counts.update(enc.tokens)
        for t in enc.tokens:
            lengths.append(len(t))
            gcs.append(gc_frac(t))

    return counts, lengths, gcs, used_regions


def tokenizer_path(root: Path, model: str, vocab: str) -> Path:
    if model == "baseline":
        return root / "tokenizer_evaluation" / "baseline_bpe" / f"vocab_{vocab}" / f"{vocab}_tokenizer.json"
    return root / "tokenizer_evaluation" / "merge_bpe" / f"vocab_{vocab}" / "merge_tokenizer_unigram_len2.json"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    intervals = reservoir_sample_bed(args.bed, args.max_regions, rng)
    fasta = pysam.FastaFile(args.fasta)

    vocabs = [v.strip() for v in args.vocabs.split(",") if v.strip()]
    rows = []

    for vocab in vocabs:
        for model in ["baseline", "merge_len2"]:
            tpath = tokenizer_path(root, model, vocab)
            tok = Tokenizer.from_file(str(tpath))
            counts, lens, gcs, used_regions = tokenize_intervals(
                fasta=fasta, intervals=intervals, tok=tok, keep_n=args.keep_n
            )
            total = sum(counts.values())
            uniq = len(counts)
            l12 = sum(1 for x in lens if x <= 2)
            l35 = sum(1 for x in lens if 3 <= x <= 5)
            l68 = sum(1 for x in lens if 6 <= x <= 8)
            l9p = sum(1 for x in lens if x >= 9)

            rows.append(
                {
                    "region": "exon",
                    "model": model,
                    "vocab": vocab,
                    "tokenizer_path": str(tpath),
                    "sampled_intervals": len(intervals),
                    "used_intervals": used_regions,
                    "total_tokens": total,
                    "unique_tokens": uniq,
                    "mean_token_len": round(statistics.mean(lens), 6) if lens else 0.0,
                    "median_token_len": round(statistics.median(lens), 6) if lens else 0.0,
                    "mean_token_gc": round(statistics.mean(gcs), 6) if gcs else 0.0,
                    "median_token_gc": round(statistics.median(gcs), 6) if gcs else 0.0,
                    "pct_len1_2": round(100 * l12 / total, 6) if total else 0.0,
                    "pct_len3_5": round(100 * l35 / total, 6) if total else 0.0,
                    "pct_len6_8": round(100 * l68 / total, 6) if total else 0.0,
                    "pct_len9plus": round(100 * l9p / total, 6) if total else 0.0,
                }
            )

            top_path = out_dir / f"top_tokens_exon_{model}_{vocab}.tsv"
            with top_path.open("w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["token", "count", "freq"])
                for token, count in counts.most_common(200):
                    w.writerow([token, count, f"{count / total:.10f}" if total else "0"])

            print(
                f"{model} {vocab}: used_intervals={used_regions:,}, total_tokens={total:,}, "
                f"mean_len={statistics.mean(lens):.3f}" if lens else f"{model} {vocab}: no tokens"
            )

    out = out_dir / "summary_exon_token_profile.tsv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
