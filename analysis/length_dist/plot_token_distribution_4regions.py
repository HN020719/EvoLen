#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import os as _os
EVOLEN_ROOT = _os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot token length-distribution comparison across promoter/enhancer/exon/intron "
            "for vocab 3072 and 5120, methods baseline and len2."
        )
    )
    p.add_argument(
        "--prom-enh-dir",
        default=f"{EVOLEN_ROOT}/token_enrichment_variants_3072_5120",
        help="Directory containing promoter-vs-enhancer all_tokens TSVs",
    )
    p.add_argument(
        "--exon-intron-dir",
        default=f"{EVOLEN_ROOT}/token_enrichment_exon_vs_intron_all4_10k",
        help="Directory containing exon-vs-intron all_tokens TSVs",
    )
    p.add_argument(
        "--out-dir",
        default=f"{EVOLEN_ROOT}/token_distribution_plots_4regions",
        help="Output directory for plots and table",
    )
    return p.parse_args()


def read_region_dist(path: Path, count_col: str) -> Dict[str, float]:
    counts = {"1-2": 0, "3-5": 0, "6-8": 0, "9+": 0}
    total = 0
    with path.open() as f:
        reader = csv.DictReader((ln for ln in f if ln.strip()), delimiter="\t")
        for row in reader:
            tok = row.get("token")
            if tok is None:
                continue
            L = len(tok)
            c = int(float(row.get(count_col, "0")))
            total += c
            if L <= 2:
                counts["1-2"] += c
            elif L <= 5:
                counts["3-5"] += c
            elif L <= 8:
                counts["6-8"] += c
            else:
                counts["9+"] += c
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: 100.0 * v / total for k, v in counts.items()}


def main() -> None:
    args = parse_args()
    prom_enh_dir = Path(args.prom_enh_dir).resolve()
    exon_intr_dir = Path(args.exon_intron_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise RuntimeError("matplotlib and numpy are required in this environment.") from e

    methods = [("baseline", "Baseline"), ("merge_len2", "Len2")]
    vocabs = ["3072", "5120"]
    bins = ["1-2", "3-5", "6-8", "9+"]
    regions = ["promoter", "enhancer", "exon", "intron"]
    region_colors = {
        "promoter": "#1f77b4",
        "enhancer": "#ff7f0e",
        "exon": "#2ca02c",
        "intron": "#d62728",
    }

    # Collect and save numeric table for manuscript/supplement
    table_rows = []
    dist_map: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for method, _label in methods:
        for vocab in vocabs:
            pe_path = prom_enh_dir / f"token_enrichment_{method}_{vocab}.all_tokens.tsv"
            ei_path = exon_intr_dir / f"token_enrichment_{method}_{vocab}.all_tokens.tsv"
            if not pe_path.exists():
                raise FileNotFoundError(f"Missing file: {pe_path}")
            if not ei_path.exists():
                raise FileNotFoundError(f"Missing file: {ei_path}")

            # promoter/enhancer
            d_prom = read_region_dist(pe_path, "count_promoter")
            d_enh = read_region_dist(pe_path, "count_enhancer")
            # exon/intron (in this run, promoter column == exon, enhancer column == intron)
            d_exon = read_region_dist(ei_path, "count_promoter")
            d_intron = read_region_dist(ei_path, "count_enhancer")

            dist_map[(method, vocab, "promoter")] = d_prom
            dist_map[(method, vocab, "enhancer")] = d_enh
            dist_map[(method, vocab, "exon")] = d_exon
            dist_map[(method, vocab, "intron")] = d_intron

            for region, d in [("promoter", d_prom), ("enhancer", d_enh), ("exon", d_exon), ("intron", d_intron)]:
                table_rows.append(
                    {
                        "method": method,
                        "vocab": vocab,
                        "region": region,
                        "pct_len1_2": d["1-2"],
                        "pct_len3_5": d["3-5"],
                        "pct_len6_8": d["6-8"],
                        "pct_len9plus": d["9+"],
                    }
                )

    tsv_path = out_dir / "token_distribution_4regions_3072_5120.tsv"
    with tsv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "vocab",
                "region",
                "pct_len1_2",
                "pct_len3_5",
                "pct_len6_8",
                "pct_len9plus",
            ],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(table_rows)

    # 4-panel plot: (baseline,len2) x (3072,5120)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    panel_order = [
        ("baseline", "3072"),
        ("merge_len2", "3072"),
        ("baseline", "5120"),
        ("merge_len2", "5120"),
    ]

    for ax, (method, vocab) in zip(axes.ravel(), panel_order):
        x = np.arange(len(bins))
        width = 0.18
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

        for region, off in zip(regions, offsets):
            vals = [dist_map[(method, vocab, region)][b] for b in bins]
            ax.bar(x + off, vals, width=width, color=region_colors[region], label=region, alpha=0.9)

        title_method = "Baseline" if method == "baseline" else "Len2"
        ax.set_title(f"{title_method} | vocab {vocab}")
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlabel("Token Length Bin")
        ax.set_ylabel("% of Tokens")
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = out_dir / "token_distribution_4regions_2vocabs_2methods.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    # also save 4 individual panels
    for method, vocab in panel_order:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(bins))
        width = 0.18
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        for region, off in zip(regions, offsets):
            vals = [dist_map[(method, vocab, region)][b] for b in bins]
            ax.bar(x + off, vals, width=width, color=region_colors[region], label=region, alpha=0.9)
        title_method = "Baseline" if method == "baseline" else "Len2"
        ax.set_title(f"{title_method} | vocab {vocab}")
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlabel("Token Length Bin")
        ax.set_ylabel("% of Tokens")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"token_distribution_{method}_{vocab}.png", dpi=220)
        plt.close(fig)

    print(f"Saved table: {tsv_path}")
    print(f"Saved combined plot: {out_png}")
    print(f"Saved 4 individual plots in: {out_dir}")


if __name__ == "__main__":
    main()
