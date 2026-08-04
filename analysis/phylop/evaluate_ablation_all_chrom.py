#!/usr/bin/env python3
"""
Evaluate ablation tokenizer variants across all chromosomes and save outputs for plotting.

Identical to evaluate_merged_uni_len2_all_chrom.py but with ablation tokenizer paths.
Only the 3 ablation variants (Baseline BPE and Len2 results already exist).
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pysam
from tokenizers import Tokenizer

import os as _os
EVOLEN_ROOT = _os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")



# =========================
# Config
# =========================
ROOT = f"{EVOLEN_ROOT}"
FASTA_PATH = os.path.join(ROOT, "hg38.fa")
BEDGRAPH_DIR = os.path.join(ROOT, "bedGraph_all")
SEGMENTS_DIR = os.path.join(ROOT, "bedGraph_all", "output")
OUT_DIR = os.path.join(ROOT, "tokenizer_evaluation", "eval_outputs_ablation")

# Keep chr1-22 + chrX/chrY by default (chrM excluded)
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

WINDOW_SIZE = 1_000_000

# The three ablation variants (§5.3). Baseline BPE and EvoLen are evaluated by
# evaluate_baseline_bpe_all_chrom.py and evaluate_merged_uni_len2_all_chrom.py.
_ABLATION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "tokenizers", "ablation",
)

VOCAB_PATHS = {
    "Ablation_no_partition": os.path.join(_ABLATION_DIR, "ablation_no_partition.json"),
    "Ablation_no_priority": os.path.join(_ABLATION_DIR, "ablation_no_priority.json"),
    "Ablation_no_length": os.path.join(_ABLATION_DIR, "ablation_no_length.json"),
}

# No optional comparison vocabs needed for ablation


# =========================
# Helper logic
# =========================
def evaluate_tokenizer_on_phyloP(tokenizer, sequences, phyloPs):
    """
    For each tokenizer, compute:
      - token_mean_scores: list of mean phyloP per token occurrence
      - token_variances:   list of variance per token occurrence
      - token_names:       list of token strings
    """

    token_means = []
    token_vars = []
    token_names = []

    total_tokens = 0

    for seq, scores in zip(sequences, phyloPs):
        # Skip if chunk is too small
        if len(seq) < 100:
            continue

        enc = tokenizer.encode(seq.upper())
        total_tokens += len(enc.ids)

        for tok, (start, end) in zip(enc.tokens, enc.offsets):
            region = scores[start:end]
            if len(region) == 0:
                continue

            m = region.mean()
            v = region.var()

            token_means.append(m)
            token_vars.append(v)
            token_names.append(tok)

    print(f"  -> Processed {total_tokens:,} tokens.")
    return {
        "mean": np.array(token_means),
        "var": np.array(token_vars),
        "token": token_names,
    }


def extract_seq_bp_by_category(fasta_path, segments, bp_scores_aligned):
    fasta = pysam.FastaFile(fasta_path)

    data = {
        "conserved": {"seqs": [], "phy": []},
        "neutral": {"seqs": [], "phy": []},
        "accelerated": {"seqs": [], "phy": []},
    }

    for _, row in segments.iterrows():
        start = int(row["genomic_start"])
        end = int(row["genomic_end"])

        seq = fasta.fetch(row["chrom"], start, end)
        phy = bp_scores_aligned[start:end]

        # skip Ns or empty slices
        if "N" in seq or len(seq) == 0:
            continue

        cat = row["category"]
        if cat not in data:
            print(f"Warning: unknown category {cat}")
            continue

        data[cat]["seqs"].append(seq)
        data[cat]["phy"].append(phy)

    return data


def aggregate_eval(df):
    return (
        df.groupby("token")
        .agg(
            mean_mean=("mean_phyloP", "mean"),
            mean_var=("var_phyloP", "mean"),
            median_mean=("mean_phyloP", "median"),
            median_var=("var_phyloP", "median"),
            count=("token", "size"),
        )
        .reset_index()
    )


def compute_summary(df, name):
    summary = {}
    summary["tokenizer"] = name

    # Conservation
    summary["mean_of_mean_phyloP"] = df["mean_mean"].mean()
    summary["median_of_mean_phyloP"] = df["mean_mean"].median()
    summary["pct_mean_phyloP_above_0"] = (df["mean_mean"] > 0).mean() * 100

    # Variance
    summary["mean_of_variance"] = df["mean_var"].mean()
    summary["median_of_variance"] = df["mean_var"].median()
    summary["pct_variance_below_0.1"] = (df["mean_var"] < 0.1).mean() * 100

    # Token frequency
    summary["num_tokens"] = len(df)
    summary["mean_token_count"] = df["count"].mean()
    summary["median_token_count"] = df["count"].median()

    return summary


def eval_out_to_df(eval_out):
    return pd.DataFrame(
        {
            "token": eval_out["token"],
            "mean_phyloP": eval_out["mean"],
            "var_phyloP": eval_out["var"],
        }
    )


# =========================
# Data loading helpers
# =========================
def load_segments_for_chrom(chrom: str) -> pd.DataFrame:
    """Load per-chrom segment file from bedGraph_all/output/chr*_phylop_segment.csv"""
    path = os.path.join(SEGMENTS_DIR, f"{chrom}_phylop_segment.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["chrom", "genomic_start", "genomic_end", "category"])

    df = pd.read_csv(path)
    required = {"chrom", "genomic_start", "genomic_end", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Segment file missing columns {missing}: {path}")

    return df[["chrom", "genomic_start", "genomic_end", "category"]]


def load_bp_scores_for_chrom(chrom: str, fasta: pysam.FastaFile) -> np.ndarray:
    """Build dense base-level phyloP array for one chromosome from per-chrom bedGraph."""
    bedgraph_path = os.path.join(BEDGRAPH_DIR, f"{chrom}.bedGraph")
    if not os.path.exists(bedgraph_path):
        raise FileNotFoundError(f"Missing bedGraph: {bedgraph_path}")

    chrom_len = fasta.get_reference_length(chrom)
    dense_scores = np.full(chrom_len, np.nan, dtype=np.float32)

    # bedGraph format: chrom start end score
    df = pd.read_csv(
        bedgraph_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "score"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64, "score": np.float32},
    )

    if len(df) == 0:
        raise ValueError(f"Empty bedGraph file: {bedgraph_path}")

    df = df[df["chrom"] == chrom]

    # Fill intervals [start, end) with score
    for s, e, v in zip(df["start"].to_numpy(), df["end"].to_numpy(), df["score"].to_numpy()):
        s0 = max(0, int(s))
        e0 = min(chrom_len, int(e))
        if e0 > s0:
            dense_scores[s0:e0] = v

    # Impute missing values with nearest as in your notebook
    s_scores = pd.Series(dense_scores)
    s_imputed = s_scores.interpolate(method="nearest", limit_direction="both")
    bp_scores_aligned = s_imputed.values.astype(np.float32)

    return bp_scores_aligned


def build_all_chrom_window_eval_data(
    fasta: pysam.FastaFile,
    chroms: List[str],
    window_size: int,
) -> Tuple[List[str], List[np.ndarray], pd.DataFrame]:
    """Create test_sequences/test_phyloP windows across all chromosomes."""
    test_sequences: List[str] = []
    test_phyloP: List[np.ndarray] = []
    meta_rows = []

    for chrom in chroms:
        print(f"Preparing windows for {chrom}")

        full_sequence = fasta.fetch(chrom)
        bp_scores_aligned = load_bp_scores_for_chrom(chrom, fasta)

        if len(full_sequence) != len(bp_scores_aligned):
            min_L = min(len(full_sequence), len(bp_scores_aligned))
            full_sequence = full_sequence[:min_L]
            bp_scores_aligned = bp_scores_aligned[:min_L]
            print(f"  Length mismatch adjusted to {min_L:,}")

        for i in range(0, len(full_sequence), window_size):
            seq = full_sequence[i : i + window_size]
            phy = bp_scores_aligned[i : i + window_size]

            # Skip low-quality windows
            if "N" in seq:
                continue

            test_sequences.append(seq)
            test_phyloP.append(phy)
            meta_rows.append({"chrom": chrom, "start": i, "end": i + len(seq), "length": len(seq)})

    meta_df = pd.DataFrame(meta_rows)
    return test_sequences, test_phyloP, meta_df


def load_tokenizers() -> Dict[str, Tokenizer]:
    tokenizers: Dict[str, Tokenizer] = {}

    # Required ablation tokenizers
    for name, path in VOCAB_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required tokenizer: {path}")
        tokenizers[name] = Tokenizer.from_file(path)

    return tokenizers


def evaluate_all_chrom_regions(tokenizers: Dict[str, Tokenizer], fasta_path: str, chroms: List[str]):
    """
    Region-based evaluation across all chromosomes.
    Keeps your original region loop: conserved/neutral/accelerated.
    """
    results = {tok_name: {"conserved": [], "neutral": [], "accelerated": []} for tok_name in tokenizers}

    fasta = pysam.FastaFile(fasta_path)

    for chrom in chroms:
        print(f"\n[Region eval] Processing {chrom}")
        segments_chr = load_segments_for_chrom(chrom)
        if len(segments_chr) == 0:
            print(f"  No segment file/rows for {chrom}, skipping region eval.")
            continue

        bp_scores_aligned = load_bp_scores_for_chrom(chrom, fasta)

        region_data = extract_seq_bp_by_category(
            fasta_path,
            segments_chr,
            bp_scores_aligned,
        )

        for tok_name, tokenizer in tokenizers.items():
            for region in ["conserved", "neutral", "accelerated"]:
                seqs = region_data[region]["seqs"]
                phys = region_data[region]["phy"]
                if len(seqs) == 0:
                    continue

                print(f"  Evaluating {tok_name} on {region} ({chrom})")
                out = evaluate_tokenizer_on_phyloP(tokenizer, seqs, phys)
                results[tok_name][region].append(out)

    return results


def merge_eval_out(eval_out_list):
    if len(eval_out_list) == 0:
        return {"token": [], "mean": np.array([]), "var": np.array([])}
    return {
        "token": sum([x["token"] for x in eval_out_list], []),
        "mean": np.concatenate([x["mean"] for x in eval_out_list]),
        "var": np.concatenate([x["var"] for x in eval_out_list]),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tokenizers...")
    tokenizers = load_tokenizers()
    print(f"Loaded {len(tokenizers)} tokenizers: {list(tokenizers.keys())}")

    fasta = pysam.FastaFile(FASTA_PATH)

    # -------------------------------------------------
    # A) Window-based evaluation data (your requested block)
    # -------------------------------------------------
    print("\nBuilding all-chrom window evaluation data...")
    test_sequences, test_phyloP, test_meta = build_all_chrom_window_eval_data(
        fasta=fasta,
        chroms=CHROMS,
        window_size=WINDOW_SIZE,
    )

    print("Num test seqs :", len(test_sequences))
    print("Num phyloP rows:", len(test_phyloP))

    eval_data_path = os.path.join(OUT_DIR, "eval_data_all_chrom.pkl")
    with open(eval_data_path, "wb") as f:
        pickle.dump(
            {
                "test_sequences": test_sequences,
                "test_phyloP": test_phyloP,
                "test_meta": test_meta,
            },
            f,
        )
    print(f"Saved {eval_data_path}")

    test_meta.to_csv(os.path.join(OUT_DIR, "eval_data_all_chrom_windows.csv"), index=False)

    # -------------------------------------------------
    # B) Tokenizer eval on all windows + save raw/agg
    # -------------------------------------------------
    summaries_window = []

    for tok_name, tok in tokenizers.items():
        print(f"\n[Window eval] {tok_name}")
        out = evaluate_tokenizer_on_phyloP(tok, test_sequences, test_phyloP)

        df_eval = eval_out_to_df(out)
        eval_csv = os.path.join(OUT_DIR, f"eval_{tok_name}.csv")
        df_eval.to_csv(eval_csv, index=False)

        print("Total rows:", len(df_eval))
        print("Unique tokens:", df_eval["token"].nunique())

        agg_df = aggregate_eval(df_eval)
        agg_csv = os.path.join(OUT_DIR, f"agg_{tok_name}.csv")
        agg_df.to_csv(agg_csv, index=False)

        summary = compute_summary(agg_df, tok_name)
        summary["scope"] = "all_chrom_window"
        summaries_window.append(summary)

    df_summary_window = pd.DataFrame(summaries_window)
    df_summary_window = df_summary_window[
        [
            "tokenizer",
            "scope",
            "mean_of_mean_phyloP",
            "median_of_mean_phyloP",
            "pct_mean_phyloP_above_0",
            "mean_of_variance",
            "median_of_variance",
            "pct_variance_below_0.1",
            "num_tokens",
            "mean_token_count",
            "median_token_count",
        ]
    ]
    df_summary_window.to_csv(os.path.join(OUT_DIR, "summary_all_chrom_window.csv"), index=False)

    # -------------------------------------------------
    # C) Region-based eval across all chromosomes
    # -------------------------------------------------
    print("\nRunning region-based evaluation across all chromosomes...")
    region_results = evaluate_all_chrom_regions(tokenizers, FASTA_PATH, CHROMS)

    summaries_region = []
    for tok_name, region_dict in region_results.items():
        for region in ["conserved", "neutral", "accelerated"]:
            merged = merge_eval_out(region_dict[region])
            df_eval = eval_out_to_df(merged)

            # Save merged region raw
            df_eval.to_csv(os.path.join(OUT_DIR, f"eval_{tok_name}_{region}.csv"), index=False)

            if len(df_eval) == 0:
                continue

            agg_df = aggregate_eval(df_eval)
            agg_df.to_csv(os.path.join(OUT_DIR, f"agg_{tok_name}_{region}.csv"), index=False)

            summary = compute_summary(agg_df, tok_name)
            summary["scope"] = "all_chrom_region"
            summary["region"] = region
            summaries_region.append(summary)

    if len(summaries_region) > 0:
        df_summary_region = pd.DataFrame(summaries_region)
        df_summary_region = df_summary_region[
            [
                "tokenizer",
                "scope",
                "region",
                "mean_of_mean_phyloP",
                "median_of_mean_phyloP",
                "pct_mean_phyloP_above_0",
                "mean_of_variance",
                "median_of_variance",
                "pct_variance_below_0.1",
                "num_tokens",
                "mean_token_count",
                "median_token_count",
            ]
        ]
        df_summary_region.to_csv(os.path.join(OUT_DIR, "summary_ablation_region.csv"), index=False)

    # -------------------------------------------------
    # D) Helper index for plotting notebooks
    # -------------------------------------------------
    file_index_rows = []
    for fn in sorted(os.listdir(OUT_DIR)):
        file_index_rows.append({"file": fn, "path": os.path.join(OUT_DIR, fn)})
    pd.DataFrame(file_index_rows).to_csv(os.path.join(OUT_DIR, "file_index.csv"), index=False)

    print("\nDone. Output directory:")
    print(OUT_DIR)
    print("Use agg_*.csv files directly for scatter/hist plots.")

def _parse_args():
    """CLI over the module-level Config block; defaults reproduce the paper runs."""
    import argparse
    p = argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fasta", default=FASTA_PATH, help="Reference FASTA")
    p.add_argument("--bedgraph-dir", default=BEDGRAPH_DIR, help="Dir of per-chrom phyloP .bedGraph")
    p.add_argument("--segments-dir", default=SEGMENTS_DIR,
                   help="Dir of {chrom}_phylop_segment.csv from process_bedgraph_all.py")
    p.add_argument("--out-dir", default=OUT_DIR, help="Where to write eval/agg/summary CSVs")
    p.add_argument("--chroms", nargs="+", default=CHROMS, help="Chromosomes to scan")
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Scan window size in bp")
    p.add_argument("--vocab-sizes", nargs="+", type=int, default=None,
                   help="Restrict to these vocab sizes (default: every configured tokenizer)")
    return p.parse_args()


if __name__ == "__main__":
    _a = _parse_args()
    FASTA_PATH = _a.fasta
    BEDGRAPH_DIR = _a.bedgraph_dir
    SEGMENTS_DIR = _a.segments_dir
    OUT_DIR = _a.out_dir
    CHROMS = list(_a.chroms)
    WINDOW_SIZE = _a.window_size
    if _a.vocab_sizes:
        _keep = tuple(str(v) for v in _a.vocab_sizes)
        VOCAB_PATHS = {k: v for k, v in VOCAB_PATHS.items() if k.endswith(_keep)}
        if not VOCAB_PATHS:
            raise SystemExit("no configured tokenizer matches --vocab-sizes %s" % (_a.vocab_sizes,))
    main()
