#!/usr/bin/env python3
import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pysam
from tokenizers import Tokenizer


BedInterval = Tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-region token importance using observed-vs-shuffled background."
        )
    )
    parser.add_argument("--fasta", required=True, help="Genome FASTA path")
    parser.add_argument("--region-bed", required=True, help="Region BED path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--region-name", required=True, help="Region label for outputs")
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix for TSV files",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=100000,
        help="Reservoir-sample up to this many regions",
    )
    parser.add_argument(
        "--num-shuffles",
        type=int,
        default=10,
        help="Number of shuffled-background replicates",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--keep-n",
        action="store_true",
        help="Keep sequences containing non-ACGT characters",
    )
    return parser.parse_args()


def is_canonical_chrom(chrom: str) -> bool:
    if chrom in {"chrX", "chrY"}:
        return True
    if not chrom.startswith("chr"):
        return False
    core = chrom[3:]
    return core.isdigit() and 1 <= int(core) <= 22


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
            if not is_canonical_chrom(chrom):
                continue
            try:
                start = int(cols[1])
                end = int(cols[2])
            except ValueError:
                continue
            if start < 0 or end <= start:
                continue
            seen += 1
            interval = (chrom, start, end)
            if len(sample) < max_regions:
                sample.append(interval)
            else:
                j = rng.randint(1, seen)
                if j <= max_regions:
                    sample[j - 1] = interval
    return sample


def fetch_sequences(
    fasta: pysam.FastaFile,
    intervals: Sequence[BedInterval],
    keep_n: bool,
) -> List[str]:
    seqs: List[str] = []
    for chrom, start, end in intervals:
        seq = fasta.fetch(chrom, start, end).upper()
        if not seq:
            continue
        if (not keep_n) and any(ch not in {"A", "C", "G", "T"} for ch in seq):
            continue
        seqs.append(seq)
    return seqs


def shuffle_seq(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def count_tokens(tokenizer: Tokenizer, seqs: Sequence[str]) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for seq in seqs:
        enc = tokenizer.encode(seq)
        if not enc.tokens:
            continue
        counts.update(enc.tokens)
        total += len(enc.tokens)
    return counts, total


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    qvals = [1.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        q = (pvals[idx] * n) / i
        prev = min(prev, q)
        qvals[idx] = min(1.0, prev)
    return qvals


def mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def stddev(vals: Sequence[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def run() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    fasta = pysam.FastaFile(args.fasta)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    intervals = reservoir_sample_bed(args.region_bed, args.max_regions, rng)
    seqs = fetch_sequences(fasta, intervals, args.keep_n)
    if not seqs:
        raise RuntimeError("No usable sequences found for region set.")

    obs_counts, obs_total = count_tokens(tokenizer, seqs)
    if obs_total == 0:
        raise RuntimeError("Tokenizer produced zero tokens on observed sequences.")

    print(f"Region: {args.region_name}")
    print(f"Sampled intervals: {len(intervals):,}")
    print(f"Usable sequences: {len(seqs):,}")
    print(f"Observed total tokens: {obs_total:,}")

    bg_counts_by_rep: List[Counter] = []
    bg_total_by_rep: List[int] = []

    for rep in range(args.num_shuffles):
        rep_rng = random.Random(args.seed + 1000 + rep)
        shuffled = [shuffle_seq(seq, rep_rng) for seq in seqs]
        counts, total = count_tokens(tokenizer, shuffled)
        bg_counts_by_rep.append(counts)
        bg_total_by_rep.append(total)
        print(f"Shuffle {rep + 1}/{args.num_shuffles}: total tokens={total:,}")

    pseudo = 0.5
    all_tokens = set(obs_counts.keys())
    for c in bg_counts_by_rep:
        all_tokens.update(c.keys())

    rows = []
    pvals = []
    for token in sorted(all_tokens):
        obs_count = obs_counts.get(token, 0)
        bg_counts = [c.get(token, 0) for c in bg_counts_by_rep]
        bg_freqs = [
            bg_counts[i] / bg_total_by_rep[i] if bg_total_by_rep[i] > 0 else 0.0
            for i in range(args.num_shuffles)
        ]

        obs_freq = obs_count / obs_total
        bg_mean_count = mean(bg_counts)
        bg_sd_count = stddev(bg_counts)
        bg_mean_freq = mean(bg_freqs)
        bg_sd_freq = stddev(bg_freqs)

        log2_enrichment = math.log2(
            (obs_freq + pseudo / obs_total)
            / (bg_mean_freq + pseudo / max(1.0, mean(bg_total_by_rep)))
        )

        if bg_sd_freq > 0:
            z = (obs_freq - bg_mean_freq) / bg_sd_freq
        else:
            z = 0.0

        ge_count = sum(1 for f in bg_freqs if f >= obs_freq)
        empirical_p = (ge_count + 1.0) / (args.num_shuffles + 1.0)

        rows.append(
            {
                "token": token,
                "count_observed": obs_count,
                "count_bg_mean": bg_mean_count,
                "count_bg_sd": bg_sd_count,
                "freq_observed": obs_freq,
                "freq_bg_mean": bg_mean_freq,
                "freq_bg_sd": bg_sd_freq,
                "log2_enrichment_observed_over_bg": log2_enrichment,
                "z_observed_vs_bg": z,
                "empirical_p_ge_bg": empirical_p,
            }
        )
        pvals.append(empirical_p)

    qvals = bh_fdr(pvals)
    for row, q in zip(rows, qvals):
        row["fdr_bh"] = q
        row["important"] = 1 if (row["log2_enrichment_observed_over_bg"] > 0 and q <= 0.05) else 0

    rows_sorted = sorted(
        rows,
        key=lambda r: (r["fdr_bh"], -r["log2_enrichment_observed_over_bg"]),
    )

    out_all = Path(f"{args.out_prefix}.all_tokens.tsv")
    with out_all.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "count_observed",
                "count_bg_mean",
                "count_bg_sd",
                "freq_observed",
                "freq_bg_mean",
                "freq_bg_sd",
                "log2_enrichment_observed_over_bg",
                "z_observed_vs_bg",
                "empirical_p_ge_bg",
                "fdr_bh",
                "important",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    out_sig = Path(f"{args.out_prefix}.important_fdr_0.05.tsv")
    important_rows = [r for r in rows_sorted if r["important"] == 1]
    with out_sig.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "count_observed",
                "count_bg_mean",
                "count_bg_sd",
                "freq_observed",
                "freq_bg_mean",
                "freq_bg_sd",
                "log2_enrichment_observed_over_bg",
                "z_observed_vs_bg",
                "empirical_p_ge_bg",
                "fdr_bh",
                "important",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(important_rows)

    print(f"Saved: {out_all}")
    print(f"Saved: {out_sig} (rows={len(important_rows):,})")


if __name__ == "__main__":
    run()
