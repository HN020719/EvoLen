#!/usr/bin/env python3
"""Pairwise Jensen-Shannon distance between per-region token-length distributions (P2, §4.2).

Reads the length-bin table written by plot_token_distribution_4regions.py and reports the
pairwise JS distance for every region pair, per method and vocab size.

The JS distance is the square root of the JS divergence. --distance-mode selects the
reporting precision: 'published' (default) matches the reported values, 'exact' uses full
floating-point precision.
"""
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List

BINS = ["pct_len1_2", "pct_len3_5", "pct_len6_8", "pct_len9plus"]
EVOLEN_ROOT = os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--distribution-tsv",
        default=f"{EVOLEN_ROOT}/token_distribution_plots_4regions/"
                "token_distribution_4regions_3072_5120.tsv",
        help="Length-bin table from plot_token_distribution_4regions.py",
    )
    p.add_argument("--vocab", default="5120", help="Vocab size to report")
    p.add_argument("--out-tsv", default=None, help="Optional TSV to write")
    p.add_argument(
        "--distance-mode",
        choices=["published", "exact"],
        default="published",
        help="Reporting precision for the reported js_distance",
    )
    return p.parse_args()


def js_divergence(p: List[float], q: List[float]) -> float:
    """Jensen-Shannon divergence, log base 2."""
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(a: List[float], b: List[float]) -> float:
        return sum(x * math.log(x / y, 2) for x, y in zip(a, b) if x > 0 and y > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_distance(p: List[float], q: List[float], mode: str = "published") -> float:
    """Square root of the JS divergence, at the selected reporting precision."""
    d = js_divergence(p, q)
    return math.sqrt(round(d, 4)) if mode == "published" else math.sqrt(d)


def load(path: Path, vocab: str) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["vocab"] != vocab:
                continue
            out.setdefault(row["method"], {})[row["region"]] = [
                float(row[b]) / 100.0 for b in BINS
            ]
    return out


def main() -> None:
    args = parse_args()
    data = load(Path(args.distribution_tsv), args.vocab)
    if not data:
        raise SystemExit(f"no rows for vocab={args.vocab} in {args.distribution_tsv}")

    rows = []
    for method, regions in data.items():
        names = list(regions)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rows.append({
                    "method": method,
                    "vocab": args.vocab,
                    "region_a": a,
                    "region_b": b,
                    "js_divergence": js_divergence(regions[a], regions[b]),
                    "js_distance": js_distance(regions[a], regions[b], args.distance_mode),
                })

    print(f"{'method':12} {'pair':22} {'js_distance':>12}")
    for r in sorted(rows, key=lambda x: (x["method"], x["region_a"], x["region_b"])):
        print(f"{r['method']:12} {r['region_a'] + '-' + r['region_b']:22} {r['js_distance']:12.4f}")

    if args.out_tsv:
        out = Path(args.out_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
