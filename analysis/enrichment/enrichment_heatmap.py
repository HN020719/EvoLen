"""
Enrichment heatmap comparing two tokenizers across genomic regions × conservation categories.

Heatmap axes:
  Rows:    promoter, enhancer, exon, intron (background)
  Columns: conserved, neutral, accelerated
  Value:   mean log2FC of token frequencies vs intron×neutral background

Two heatmaps side by side: baseline BPE vs merge_len2 BPE (vocab 5120)
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer

import os as _os

# Working root: hg38, phyloP segments, generated outputs.
EVOLEN_ROOT = _os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")
# Extracted EvoLenTokenizer/analysis-data bundle (region BEDs, cCRE classes, motifs).
EVOLEN_DATA = _os.environ.get(
    "EVOLEN_DATA", _os.path.expanduser("~/evolen_analysis/evolen_analysis_data")
)
# Tokenizers ship with the repo, two levels above analysis/enrichment/.
_REPO = Path(__file__).resolve().parents[2]


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(f"{EVOLEN_ROOT}")
DATA        = Path(EVOLEN_DATA)
PHYLOP_DIR  = Path(_os.environ.get("EVOLEN_SEGMENTS", str(ROOT / "bedGraph_all/output")))
GENOME      = Path(_os.environ.get("EVOLEN_FASTA", str(ROOT / "hg38.fa")))
OUTDIR      = Path(_os.environ.get("EVOLEN_OUTDIR", str(ROOT / "enrichment_analysis")))

REGION_BEDS = {
    "promoter": DATA / "region_beds/source/promoters_2kb.clean.merged.bed",
    "enhancer": DATA / "region_beds/source/enhancers_dels.clean.merged.bed",
    "exon":     DATA / "region_beds/source/exon.clean.merged.bed",
    "intron":   DATA / "region_beds/source/intron.clean.merged.bed",
}

TOKENIZER_PATHS = {
    "baseline":   _REPO / "assets/tokenizers/5120_tokenizer.json",
    "merge_len2": _REPO / "assets/tokenizers/merge_tokenizer_unigram_len2.json",
}

REGIONS       = ["promoter", "enhancer", "exon", "intron"]
CONS_CATS     = ["conserved", "neutral", "accelerated"]
BACKGROUND    = ("intron", "neutral")   # reference cell for log2FC


# ── Step 1: Merge phyloP CSVs → one BED per conservation category ─────────────
def build_conservation_beds(phylop_dir: Path, outdir: Path) -> dict[str, Path]:
    print("Building conservation BED files...")
    dfs = [pd.read_csv(f) for f in sorted(phylop_dir.glob("chr*_phylop_segment.csv"))]
    df = pd.concat(dfs, ignore_index=True)

    beds = {}
    for cat in CONS_CATS:
        sub = df[df["category"] == cat][["chrom", "genomic_start", "genomic_end"]]
        bed_path = outdir / f"conservation_{cat}.bed"
        sub.to_csv(bed_path, sep="\t", header=False, index=False)
        beds[cat] = bed_path
        print(f"  {cat}: {len(sub):,} intervals")
    return beds


# ── Step 2: Intersect region BED × conservation BED ──────────────────────────
def intersect(region_bed: Path, cons_bed: Path, out_bed: Path):
    if out_bed.exists() and out_bed.stat().st_size > 0:
        return  # skip if already done
    subprocess.run(
        f"bedtools intersect -a {region_bed} -b {cons_bed} | "
        f"bedtools sort -i - | bedtools merge -i - > {out_bed}",
        shell=True, check=True
    )


# ── Step 3: Extract sequences from genome ────────────────────────────────────
def get_fasta(bed: Path, genome: Path, out_fa: Path):
    if out_fa.exists() and out_fa.stat().st_size > 0:
        return
    subprocess.run(
        f"bedtools getfasta -fi {genome} -bed {bed} -fo {out_fa}",
        shell=True, check=True
    )


# ── Step 4: Tokenize a FASTA file → token Counter ────────────────────────────
def tokenize_fasta(fasta: Path, tok: Tokenizer, batch_size: int = 2000) -> Counter:
    """Read all sequences, then encode in batches using encode_batch() for speed."""
    counts = Counter()
    sequences = []
    seq_parts = []

    with open(fasta) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_parts:
                    sequences.append("".join(seq_parts))
                    seq_parts = []
            else:
                seq_parts.append(line.upper().replace("N", ""))
    if seq_parts:
        sequences.append("".join(seq_parts))

    # Remove empty sequences
    sequences = [s for s in sequences if s]

    # Encode in batches
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        encodings = tok.encode_batch(batch)
        for enc in encodings:
            counts.update(enc.tokens)

    return counts


# ── Step 5: Compute mean log2FC vs background bin ────────────────────────────
def mean_log2fc(bin_counts: Counter, bg_counts: Counter) -> float:
    """Mean log2(fold-change) of per-token frequencies vs background."""
    bin_total = sum(bin_counts.values()) or 1
    bg_total  = sum(bg_counts.values()) or 1
    vocab     = set(bin_counts) | set(bg_counts)
    pseudo    = 0.5  # add-0.5 pseudocount

    log2fcs = []
    for tok in vocab:
        bin_freq = (bin_counts.get(tok, 0) + pseudo) / (bin_total + pseudo * len(vocab))
        bg_freq  = (bg_counts.get(tok,  0) + pseudo) / (bg_total  + pseudo * len(vocab))
        log2fcs.append(np.log2(bin_freq / bg_freq))
    return float(np.mean(log2fcs))


# ── Plot-only helper ─────────────────────────────────────────────────────────
def load_cached_matrices() -> dict[str, pd.DataFrame]:
    """Load pre-computed enrichment matrices from CSVs."""
    results = {}
    for tok_name in TOKENIZER_PATHS:
        csv_path = OUTDIR / f"enrichment_matrix_{tok_name}.csv"
        results[tok_name] = pd.read_csv(csv_path, index_col=0)
        print(f"Loaded {csv_path}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data pipeline; re-plot from cached CSVs")
    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)

    if args.plot_only:
        results = load_cached_matrices()
    else:
        beds_dir = OUTDIR / "beds"
        fasta_dir = OUTDIR / "fastas"
        beds_dir.mkdir(exist_ok=True)
        fasta_dir.mkdir(exist_ok=True)

        # Step 1: conservation BEDs
        cons_beds = build_conservation_beds(PHYLOP_DIR, beds_dir)

        # Step 2 & 3: intersect and extract sequences for all 12 bins
        print("Intersecting regions with conservation categories...")
        bin_fastas = {}   # (region, cat) -> Path
        for region, region_bed in REGION_BEDS.items():
            for cat, cons_bed in cons_beds.items():
                tag      = f"{region}_{cat}"
                bin_bed  = beds_dir  / f"{tag}.bed"
                bin_fa   = fasta_dir / f"{tag}.fa"
                intersect(region_bed, cons_bed, bin_bed)
                get_fasta(bin_bed, GENOME, bin_fa)
                bin_fastas[(region, cat)] = bin_fa
                n = sum(1 for l in open(bin_bed) if l.strip()) if bin_bed.exists() else 0
                print(f"  {tag}: {n} intervals")

        # Step 4 & 5: tokenize and compute enrichment matrix per tokenizer
        results = {}   # tokenizer_name -> DataFrame (regions × cons_cats)

        for tok_name, tok_path in TOKENIZER_PATHS.items():
            print(f"\nTokenizing with {tok_name}...")
            tok = Tokenizer.from_file(str(tok_path))

            # Tokenize all 12 bins
            counts = {}
            for (region, cat), fa in bin_fastas.items():
                print(f"  {region} × {cat} ...", end=" ", flush=True)
                counts[(region, cat)] = tokenize_fasta(fa, tok)
                print(f"{sum(counts[(region, cat)].values()):,} tokens")

            # Background = intron × neutral
            bg = counts[BACKGROUND]

            # Build 4×3 matrix
            matrix = np.zeros((len(REGIONS), len(CONS_CATS)))
            for i, region in enumerate(REGIONS):
                for j, cat in enumerate(CONS_CATS):
                    matrix[i, j] = mean_log2fc(counts[(region, cat)], bg)

            results[tok_name] = pd.DataFrame(matrix, index=REGIONS, columns=CONS_CATS)
            print(f"  Matrix:\n{results[tok_name].round(3)}")

        # Save matrices as CSVs
        for tok_name, df in results.items():
            df.to_csv(OUTDIR / f"enrichment_matrix_{tok_name}.csv")

    # Plot heatmaps in a grid adapting to number of tokenizers
    n_tok = len(results)
    ncols = 3
    nrows = (n_tok + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), constrained_layout=True)
    axes_flat = axes.ravel()

    vmin = min(df.values.min() for df in results.values())
    vmax = max(df.values.max() for df in results.values())

    from matplotlib.colors import LinearSegmentedColormap
    cmap_custom = LinearSegmentedColormap.from_list(
        "bpe_evo", ["#F8A5A5", "white", "#7EDCC5"])

    title_map = {
        "baseline": "Baseline BPE",
        "merge_len2": "EvoLen (5120)",
    }

    for idx, (tok_name, df) in enumerate(results.items()):
        ax = axes_flat[idx]
        show_yticklabels = (idx % ncols == 0)
        show_cbar = (idx % ncols == ncols - 1) or (idx == n_tok - 1)
        sns.heatmap(
            df,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 13},
            cmap=cmap_custom,
            center=0,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            cbar=show_cbar,
            yticklabels=show_yticklabels,
        )
        ax.set_title(title_map.get(tok_name, tok_name), fontsize=16,
                     fontweight="bold")
        ax.set_xlabel("Conservation category", fontsize=14)
        if show_yticklabels:
            ax.set_ylabel("Genomic region", fontsize=14)
            ax.tick_params(axis="y", labelsize=13, rotation=0)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=13)

    # Hide unused axes
    for idx in range(n_tok, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Mean log2FC of token frequencies vs intron × neutral background",
        fontsize=20,
    )
    out_fig = OUTDIR / "enrichment_heatmap.pdf"
    plt.savefig(out_fig, bbox_inches="tight", dpi=150)
    plt.savefig(str(out_fig).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"\nSaved: {out_fig}")


if __name__ == "__main__":
    main()
