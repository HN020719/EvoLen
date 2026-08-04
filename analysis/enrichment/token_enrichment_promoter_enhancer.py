#!/usr/bin/env python3
import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pysam
from tokenizers import Tokenizer


BedInterval = Tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token enrichment analysis between promoter and enhancer regions."
    )
    parser.add_argument("--fasta", required=True, help="Genome FASTA path (e.g., hg38.fa)")
    parser.add_argument("--promoter-bed", required=True, help="Promoter BED (3+ columns)")
    parser.add_argument("--enhancer-bed", required=True, help="Enhancer BED (3+ columns)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument(
        "--out-prefix",
        default="token_enrichment_promoter_vs_enhancer",
        help="Output prefix for TSV files",
    )
    parser.add_argument(
        "--max-regions-per-set",
        type=int,
        default=100000,
        help="Reservoir-sample up to this many intervals per set (default: 100000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--keep-n",
        action="store_true",
        help="Keep sequences containing N (default: drop sequences with non-ACGT)",
    )
    return parser.parse_args()


def is_canonical_chrom(chrom: str) -> bool:
    if chrom in {"chrX", "chrY"}:
        return True
    if not chrom.startswith("chr"):
        return False
    core = chrom[3:]
    if not core.isdigit():
        return False
    num = int(core)
    return 1 <= num <= 22


def reservoir_sample_bed(path: str, max_regions: int, rng: random.Random) -> List[BedInterval]:
    sample: List[BedInterval] = []
    seen = 0
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chrom = fields[0]
            if not is_canonical_chrom(chrom):
                continue
            try:
                start = int(fields[1])
                end = int(fields[2])
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


def tokenize_intervals(
    fasta: pysam.FastaFile,
    intervals: Sequence[BedInterval],
    tokenizer: Tokenizer,
    keep_n: bool,
) -> Tuple[Counter, int, int]:
    counts: Counter = Counter()
    total_tokens = 0
    used_regions = 0

    for chrom, start, end in intervals:
        seq = fasta.fetch(chrom, start, end).upper()
        if not seq:
            continue
        if (not keep_n) and any(ch not in {"A", "C", "G", "T"} for ch in seq):
            continue
        enc = tokenizer.encode(seq)
        if len(enc.tokens) == 0:
            continue
        counts.update(enc.tokens)
        total_tokens += len(enc.tokens)
        used_regions += 1

    return counts, total_tokens, used_regions


def normal_p_two_sided_from_z(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    qvals = [1.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        q = (pvals[idx] * n) / i
        prev = min(prev, q)
        qvals[idx] = min(1.0, prev)
    return qvals


def run() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    fasta = pysam.FastaFile(args.fasta)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    promoters = reservoir_sample_bed(args.promoter_bed, args.max_regions_per_set, rng)
    enhancers = reservoir_sample_bed(args.enhancer_bed, args.max_regions_per_set, rng)

    print(f"Sampled promoters: {len(promoters):,}")
    print(f"Sampled enhancers: {len(enhancers):,}")

    prom_counts, prom_total, prom_used = tokenize_intervals(
        fasta=fasta, intervals=promoters, tokenizer=tokenizer, keep_n=args.keep_n
    )
    enh_counts, enh_total, enh_used = tokenize_intervals(
        fasta=fasta, intervals=enhancers, tokenizer=tokenizer, keep_n=args.keep_n
    )

    if prom_total == 0 or enh_total == 0:
        raise RuntimeError(
            f"Zero tokens found (prom_total={prom_total}, enh_total={enh_total}). "
            "Try --keep-n or check tokenizer/FASTA/BED compatibility."
        )

    print(f"Used promoter regions: {prom_used:,}, total promoter tokens: {prom_total:,}")
    print(f"Used enhancer regions: {enh_used:,}, total enhancer tokens: {enh_total:,}")

    pseudo = 0.5
    tokens = sorted(set(prom_counts.keys()) | set(enh_counts.keys()))
    rows = []
    pvals = []

    for tok in tokens:
        cp = prom_counts.get(tok, 0)
        ce = enh_counts.get(tok, 0)

        fp = cp / prom_total
        fe = ce / enh_total
        log2fc = math.log2((fp + pseudo / prom_total) / (fe + pseudo / enh_total))

        # Two-proportion z-test
        pooled = (cp + ce) / (prom_total + enh_total)
        var = pooled * (1.0 - pooled) * (1.0 / prom_total + 1.0 / enh_total)
        if var > 0:
            z = (fp - fe) / math.sqrt(var)
            pval = normal_p_two_sided_from_z(z)
        else:
            z = 0.0
            pval = 1.0

        side = "promoter" if log2fc > 0 else "enhancer" if log2fc < 0 else "equal"
        rows.append(
            {
                "token": tok,
                "count_promoter": cp,
                "count_enhancer": ce,
                "freq_promoter": fp,
                "freq_enhancer": fe,
                "log2fc_promoter_over_enhancer": log2fc,
                "z": z,
                "p_value": pval,
                "enriched_in": side,
            }
        )
        pvals.append(pval)

    qvals = bh_fdr(pvals)
    for row, q in zip(rows, qvals):
        row["fdr_bh"] = q

    rows_sorted = sorted(rows, key=lambda r: r["fdr_bh"])

    out_all = Path(f"{args.out_prefix}.all_tokens.tsv")
    with out_all.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "count_promoter",
                "count_enhancer",
                "freq_promoter",
                "freq_enhancer",
                "log2fc_promoter_over_enhancer",
                "z",
                "p_value",
                "fdr_bh",
                "enriched_in",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    out_sig = Path(f"{args.out_prefix}.sig_fdr_0.05.tsv")
    sig_rows = [r for r in rows_sorted if r["fdr_bh"] <= 0.05]
    with out_sig.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "count_promoter",
                "count_enhancer",
                "freq_promoter",
                "freq_enhancer",
                "log2fc_promoter_over_enhancer",
                "z",
                "p_value",
                "fdr_bh",
                "enriched_in",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(sig_rows)

    print(f"Saved: {out_all}")
    print(f"Saved: {out_sig} (rows={len(sig_rows):,})")


if __name__ == "__main__":
    run()
