#!/usr/bin/env python3
import argparse
import os
import glob
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm

COLUMNS = ["chrom", "start", "end", "score"]
DTYPES = {"chrom": str, "start": np.int64, "end": np.int64, "score": np.float64}

def merge_adjacent_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges adjacent rows into contiguous blocks. Assumes input DataFrame
    is sorted by 'chrom' and 'genomic_start' (which it should be).
    """
    if df.empty:
        return df.copy()

    # Calculate the end position of the previous bin
    df['prev_genomic_end'] = df['genomic_end'].shift(1)
    
    # Identify where a new contiguous block starts:
    # A new block starts if:
    # 1. It's the very first row (shifted end is NaN)
    # 2. OR the current genomic_start does NOT equal the previous genomic_end
    df['block_start'] = (df['genomic_start'] != df['prev_genomic_end']) | df['prev_genomic_end'].isna()
    
    # Create a unique ID for each contiguous block
    df['block_id'] = df['block_start'].cumsum()
    
    # Group by the block ID and aggregate the coordinates
    merged_df = df.groupby(['chrom', 'block_id']).agg(
        genomic_start=('genomic_start', 'first'),
        genomic_end=('genomic_end', 'last')
    ).reset_index()
    
    # Cleanup
    merged_df.drop(columns=['block_id'], inplace=True)
    
    return merged_df

def process_bedgraph(file_path, out_path, bin_size=100, p_value=1e-1, chunk_size=1_000_000):
    # Load in chunks to avoid RAM blow-up
    all_chunks = []
    skipped = []

    def bad_line_handler(bad_line):
        skipped.append(" ".join(map(str, bad_line)))
        return None

    print(f"Loading data in chunks of {chunk_size:,} lines...")
    try:
        chunk_iter = pd.read_csv(
            file_path,
            sep="\t",
            names=COLUMNS,
            dtype=DTYPES,
            chunksize=chunk_size,
            engine="python",
            on_bad_lines=bad_line_handler,
        )

        for i, chunk in enumerate(chunk_iter):
            all_chunks.append(chunk)
            print(f"Processed Chunk {i+1}: Shape {chunk.shape}")

        if not all_chunks:
            logging.warning("Empty or unreadable: %s", file_path)
            return

        df_full = pd.concat(all_chunks, ignore_index=True)
        print(f"\nFinal DataFrame loaded successfully. Total rows: {len(df_full):,}")
    except Exception as e:
        print(f"An error occurred during chunk processing: {e}")
        return

    print("\n--- Summary of Skipped Lines ---")
    if skipped:
        print(f"Total lines skipped and stored: {len(skipped):,}")
        print("Preview of the first 5 skipped lines:")
        for line in skipped[:5]:
            print(f"  {line}")
    else:
        print("No lines were skipped.")

    # Bin and average
    df_full["bin"] = (df_full["start"] / bin_size).astype(int)
    binned = df_full.groupby("bin")["score"].mean().reset_index()
    binned.rename(columns={"score": "binned_score"}, inplace=True)

    binned["genomic_start"] = binned["bin"] * bin_size
    chrom = df_full["chrom"].iloc[0]
    binned["chrom"] = chrom
    print(binned.head())

    # Z-score thresholds
    mu = binned["binned_score"].mean()
    sigma = binned["binned_score"].std()

    p_upper = 1 - (p_value / 2)
    z_cutoff = norm.ppf(p_upper, loc=0, scale=1)
    score_cutoff_upper = mu + z_cutoff * sigma
    score_cutoff_lower = mu - z_cutoff * sigma

    print(f"Mean Binned Score (Background, μ): {mu:.3f}")
    print(f"Standard Deviation (Background, σ): {sigma:.3f}")
    print(f"Z-score Cutoff for Two-Tailed P < {p_value}: ±{z_cutoff:.3f}")
    print(f"Absolute Score Cutoffs: Lower={score_cutoff_lower:.3f}, Upper={score_cutoff_upper:.3f}")

    def categorize(score):
        if score < score_cutoff_lower:
            return "accelerated"
        elif score > score_cutoff_upper:
            return "conserved"
        else:
            return "neutral"

    binned["category"] = binned["binned_score"].apply(categorize)

    # If you want to keep a helper boolean for important:
    binned["is_important"] = binned["category"] == "conserved"

    # Count the total number of bins analyzed
    total_bins = len(binned)

    # Count the number of bins considered "important"
    important_bins = binned['is_important'].sum()

    # Calculate the percentage
    percentage_important = (important_bins / total_bins) * 100

    print(f"Total bins analyzed: {total_bins}")
    print(f"Number of bins considered 'important': {important_bins}")
    print(f"Percentage of sequence considered important: {percentage_important:.4f}%")

    # Split and merge
    accelerated = binned[binned["category"] == "accelerated"].copy()
    neutral = binned[binned["category"] == "neutral"].copy()
    conserved = binned[binned["category"] == "conserved"].copy()

    # The end position is the start position plus the maxlen/bin size
    for df in [accelerated, neutral, conserved]:
        df["genomic_end"] = df["genomic_start"] + bin_size

    final_acc = merge_adjacent_regions(accelerated)
    final_neu = merge_adjacent_regions(neutral)
    final_con = merge_adjacent_regions(conserved)

    print("--- FINAL MERGED accelerated REGIONS ---")
    print(f"Original Bins: {len(accelerated)} -> Merged Regions: {len(final_acc)}")
    print(final_acc.head())

    print("\n--- FINAL MERGED neutral REGIONS ---")
    print(f"Original Bins: {len(neutral)} -> Merged Regions: {len(final_neu)}")
    print(final_neu.head())

    print("\n--- FINAL MERGED conserved REGIONS ---")
    print(f"Original Bins: {len(conserved)} -> Merged Regions: {len(final_con)}")
    print(final_con.head())

    final_acc["category"] = "accelerated"
    final_neu["category"] = "neutral"
    final_con["category"] = "conserved"

    df_combined = pd.concat([final_acc, final_neu, final_con], ignore_index=True)
    df_combined = df_combined.sort_values(by="genomic_start").reset_index(drop=True)
    print(df_combined.head())

    df_combined.to_csv(out_path, index=False)
    logging.info("Wrote %s (skipped lines: %s)", out_path, len(skipped))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedgraph_dir", required=True, help="Dir with bedGraph files")
    parser.add_argument("--out_dir", required=True, help="Dir to write per-chrom segment CSVs")
    parser.add_argument("--bin_size", type=int, default=100)
    parser.add_argument("--p_value", type=float, default=1e-1)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--chrom", type=str, default=None, help="Process only this chromosome (e.g., chr1)")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    log_path = args.log_file or os.path.join(args.out_dir, "process_bedgraph_all.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting with bedgraph_dir=%s out_dir=%s chrom=%s", args.bedgraph_dir, args.out_dir, args.chrom)

    pattern = "chr*.bedGraph" if not args.chrom else f"{args.chrom}.bedGraph"
    for file_path in sorted(glob.glob(os.path.join(args.bedgraph_dir, pattern))):
        chrom = os.path.basename(file_path).split(".")[0]
        out_path = os.path.join(args.out_dir, f"{chrom}_phylop_segment.csv")
        logging.info("Processing %s -> %s", file_path, out_path)
        process_bedgraph(
            file_path,
            out_path,
            bin_size=args.bin_size,
            p_value=args.p_value,
            chunk_size=args.chunk_size,
        )

if __name__ == "__main__":
    main()
