#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import os as _os
EVOLEN_ROOT = _os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fold-change heatmaps (len2 / baseline) for region-by-length-bin token distributions."
        )
    )
    parser.add_argument(
        "--distribution-tsv",
        default=f"{EVOLEN_ROOT}/token_distribution_plots_4regions/token_distribution_4regions_3072_5120.tsv",
        help="TSV from plot_token_distribution_4regions.py",
    )
    parser.add_argument(
        "--vocab",
        default="5120",
        help="Vocab size to plot",
    )
    parser.add_argument(
        "--out-dir",
        default=f"{EVOLEN_ROOT}/token_distribution_plots_4regions",
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist_path = Path(args.distribution_tsv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise RuntimeError("matplotlib and numpy are required for plotting.") from e

    bins = ["pct_len1_2", "pct_len3_5", "pct_len6_8", "pct_len9plus"]
    bin_labels = ["1-2", "3-5", "6-8", "9+"]
    all_regions = ["promoter", "enhancer", "exon", "intron"]
    trio_regions = ["promoter", "enhancer", "exon"]

    baseline = {}
    len2 = {}
    with dist_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["vocab"] != args.vocab:
                continue
            vals = [float(row[b]) for b in bins]
            if row["method"] == "baseline":
                baseline[row["region"]] = vals
            elif row["method"] == "merge_len2":
                len2[row["region"]] = vals

    for region in all_regions:
        if region not in baseline or region not in len2:
            raise ValueError(f"Missing region {region} for vocab {args.vocab}")

    def build_matrix(regions):
        data = []
        for region in regions:
            row = []
            for i in range(len(bins)):
                b = baseline[region][i]
                m = len2[region][i]
                # baseline as 1.0 reference; assume no zero bins in current data
                row.append(m / b if b != 0 else float("nan"))
            data.append(row)
        return np.array(data)

    full_mat = build_matrix(all_regions)
    trio_mat = build_matrix(trio_regions)

    # Save numeric TSV
    out_tsv = out_dir / f"fold_change_heatmap_vocab_{args.vocab}.tsv"
    with out_tsv.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["region"] + bin_labels)
        for region, vals in zip(all_regions, full_mat):
            writer.writerow([region] + [f"{v:.6f}" for v in vals])

    def draw_heatmap(mat, regions, title, out_png):
        fig, ax = plt.subplots(figsize=(7, 4 + 0.6 * len(regions)))
        # Custom diverging colormap: BPE-dominant (#D63031) ↔ white ↔ evoToken-dominant (#00B894)
        from matplotlib.colors import LinearSegmentedColormap
        cmap_custom = LinearSegmentedColormap.from_list(
            "bpe_evo", ["#D63031", "white", "#00B894"])
        im = ax.imshow(mat, aspect="auto", cmap=cmap_custom, vmin=0.5, vmax=1.5)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions)
        ax.set_title(title)
        for i in range(len(regions)):
            for j in range(len(bin_labels)):
                ax.text(j, i, f"{mat[i, j]:.2f}x", ha="center", va="center", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Fold change (evoToken / Baseline BPE)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)

    draw_heatmap(
        full_mat,
        all_regions,
        f"Fold-change heatmap | vocab {args.vocab} | len2 / baseline",
        out_dir / f"fold_change_heatmap_vocab_{args.vocab}.png",
    )

    draw_heatmap(
        trio_mat,
        trio_regions,
        f"Fold-change heatmap (trio) | vocab {args.vocab} | len2 / baseline",
        out_dir / f"fold_change_heatmap_trio_vocab_{args.vocab}.png",
    )

    print(f"Saved fold-change table: {out_tsv}")
    print(f"Saved full heatmap: {out_dir / f'fold_change_heatmap_vocab_{args.vocab}.png'}")
    print(f"Saved trio heatmap: {out_dir / f'fold_change_heatmap_trio_vocab_{args.vocab}.png'}")


if __name__ == "__main__":
    main()
