#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot volcano plots for merge_len2 vs baseline token usage in "
            "enhancer/promoter for vocab sizes 3072 and 5120."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory with token_enrichment_<model>_<vocab>.all_tokens.tsv files",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots (default: <results-dir>/plots)",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="FDR threshold for significance coloring",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=15,
        help="Number of top significant tokens to annotate per plot",
    )
    parser.add_argument(
        "--vocabs",
        default="2048,3072,4096,5120",
        help="Comma-separated vocab sizes to plot (e.g., 3072,5120)",
    )
    return parser.parse_args()


def read_counts(path: Path, region_col: str) -> Tuple[Dict[str, int], int]:
    counts: Dict[str, int] = {}
    total = 0
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tok = row["token"]
            c = int(float(row[region_col]))
            counts[tok] = c
            total += c
    return counts, total


def normal_p_two_sided_from_z(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def bh_fdr(pvals: List[float]) -> List[float]:
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


def build_volcano_rows(
    merge_counts: Dict[str, int],
    merge_total: int,
    base_counts: Dict[str, int],
    base_total: int,
) -> List[Dict[str, float]]:
    pseudo = 0.5
    rows: List[Dict[str, float]] = []
    pvals: List[float] = []
    all_tokens = set(merge_counts) | set(base_counts)

    for tok in all_tokens:
        cm = merge_counts.get(tok, 0)
        cb = base_counts.get(tok, 0)

        fm = cm / merge_total if merge_total > 0 else 0.0
        fb = cb / base_total if base_total > 0 else 0.0
        log2fc = math.log2((fm + pseudo / merge_total) / (fb + pseudo / base_total))

        pooled = (cm + cb) / (merge_total + base_total)
        var = pooled * (1.0 - pooled) * (1.0 / merge_total + 1.0 / base_total)
        if var > 0:
            z = (fm - fb) / math.sqrt(var)
            p = normal_p_two_sided_from_z(z)
        else:
            z = 0.0
            p = 1.0

        rows.append(
            {
                "token": tok,
                "count_merge": cm,
                "count_baseline": cb,
                "freq_merge": fm,
                "freq_baseline": fb,
                "log2fc_merge_over_baseline": log2fc,
                "z": z,
                "p_value": p,
            }
        )
        pvals.append(p)

    qvals = bh_fdr(pvals)
    for row, q in zip(rows, qvals):
        row["fdr_bh"] = q
        row["neglog10_fdr"] = -math.log10(max(q, 1e-300))
    return rows


def save_table(rows: List[Dict[str, float]], out_path: Path) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r["fdr_bh"], -abs(r["log2fc_merge_over_baseline"])))
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "count_merge",
                "count_baseline",
                "freq_merge",
                "freq_baseline",
                "log2fc_merge_over_baseline",
                "z",
                "p_value",
                "fdr_bh",
                "neglog10_fdr",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows_sorted)


def plot_one(ax, rows: List[Dict[str, float]], title: str, fdr_threshold: float, top_labels: int) -> None:
    sig_y = -math.log10(fdr_threshold)
    x = [r["log2fc_merge_over_baseline"] for r in rows]
    y = [r["neglog10_fdr"] for r in rows]

    x_ns = []
    y_ns = []
    x_m = []
    y_m = []
    x_b = []
    y_b = []
    for r in rows:
        sig = r["fdr_bh"] <= fdr_threshold
        if not sig:
            x_ns.append(r["log2fc_merge_over_baseline"])
            y_ns.append(r["neglog10_fdr"])
        elif r["log2fc_merge_over_baseline"] >= 0:
            x_m.append(r["log2fc_merge_over_baseline"])
            y_m.append(r["neglog10_fdr"])
        else:
            x_b.append(r["log2fc_merge_over_baseline"])
            y_b.append(r["neglog10_fdr"])

    ax.scatter(x_ns, y_ns, s=8, c="#BFBFBF", alpha=0.5, linewidths=0)
    ax.scatter(x_m, y_m, s=10, c="#D62728", alpha=0.8, linewidths=0, label="merge_len2-enriched")
    ax.scatter(x_b, y_b, s=10, c="#1F77B4", alpha=0.8, linewidths=0, label="baseline-enriched")
    ax.axhline(sig_y, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1)

    # Annotate most informative points (strong sig + strong effect)
    ranked = sorted(
        [r for r in rows if r["fdr_bh"] <= fdr_threshold],
        key=lambda r: (r["fdr_bh"], -abs(r["log2fc_merge_over_baseline"])),
    )[:top_labels]
    for r in ranked:
        ax.text(
            r["log2fc_merge_over_baseline"],
            r["neglog10_fdr"],
            r["token"],
            fontsize=7,
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("log2FC (merge_len2 / baseline)")
    ax.set_ylabel("-log10(FDR)")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (results_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it in your env: "
            "conda install matplotlib"
        ) from e

    allowed = {"2048", "3072", "4096", "5120"}
    vocabs = [v.strip() for v in args.vocabs.split(",") if v.strip()]
    if not vocabs:
        raise ValueError("No vocab provided in --vocabs")
    bad = [v for v in vocabs if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported vocab(s): {bad}. Allowed: {sorted(allowed)}")

    configs = []
    for vocab in vocabs:
        configs.append((vocab, "enhancer", "count_enhancer"))
        configs.append((vocab, "promoter", "count_promoter"))

    panel_rows: Dict[Tuple[str, str], List[Dict[str, float]]] = {}

    for vocab, region_name, col in configs:
        merge_file = results_dir / f"token_enrichment_merge_len2_{vocab}.all_tokens.tsv"
        base_file = results_dir / f"token_enrichment_baseline_{vocab}.all_tokens.tsv"
        if not merge_file.exists() or not base_file.exists():
            raise FileNotFoundError(
                f"Missing required files for vocab {vocab} region {region_name}: "
                f"{merge_file} / {base_file}"
            )

        merge_counts, merge_total = read_counts(merge_file, col)
        base_counts, base_total = read_counts(base_file, col)
        rows = build_volcano_rows(merge_counts, merge_total, base_counts, base_total)
        panel_rows[(vocab, region_name)] = rows

        table_out = out_dir / f"volcano_data_{vocab}_{region_name}.tsv"
        save_table(rows, table_out)

        fig, ax = plt.subplots(figsize=(7, 5))
        plot_one(
            ax=ax,
            rows=rows,
            title=f"{vocab} {region_name}: merge_len2 vs baseline",
            fdr_threshold=args.fdr_threshold,
            top_labels=args.top_labels,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"volcano_{vocab}_{region_name}.png", dpi=200)
        plt.close(fig)

    # Combined panel (rows = vocabs, cols = enhancer/promoter)
    nrows = len(vocabs)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    if nrows == 1:
        axes = [axes]
    order = []
    for vocab in vocabs:
        order.append((vocab, "enhancer"))
        order.append((vocab, "promoter"))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [ax for row in axes for ax in row]
    for ax, (vocab, region_name) in zip(flat_axes, order):
        plot_one(
            ax=ax,
            rows=panel_rows[(vocab, region_name)],
            title=f"{vocab} {region_name}: merge_len2 vs baseline",
            fdr_threshold=args.fdr_threshold,
            top_labels=args.top_labels,
        )
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "volcano_panel_merge_len2_vs_baseline.png", dpi=220)
    plt.close(fig)

    print(f"Saved plots and data to: {out_dir}")


if __name__ == "__main__":
    main()
