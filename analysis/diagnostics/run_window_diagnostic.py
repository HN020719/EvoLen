#!/usr/bin/env python3
"""phyloP window-size / bin-stability / heterogeneity / boundary diagnostics on chr1.

Inputs:
  - /data/n5huang/dna_token/hg38.phyloP20way.bw  (phyloP 20-way)
  - chr1 only

Outputs (under tokenizer_evaluation/):
  - phylop_window_smoothing_chr1.csv    (Analysis 1)
  - phylop_category_stability_chr1.csv  (Analysis 2 — fractions + agreement vs 100bp)
  - phylop_within_bin_heterogeneity_chr1.csv  (Analysis 3, at W=100)
  - phylop_boundary_cross_prob.csv      (Analysis 4 — purely analytic)
"""
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pyBigWig

import os as _os
EVOLEN_ROOT = _os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")


OUT = Path(f"{EVOLEN_ROOT}/tokenizer_evaluation")
OUT.mkdir(parents=True, exist_ok=True)
BW_PATH = f"{EVOLEN_ROOT}/hg38.phyloP20way.bw"
CHROM = "chr1"
WINDOW_SIZES = [5, 10, 25, 50, 100, 200, 500]
Z_THRESH = 1.645  # paper: two-tailed p<0.1
REFERENCE_W = 100


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_chr_phylop():
    log(f"Opening {BW_PATH}")
    bw = pyBigWig.open(BW_PATH)
    length = bw.chroms(CHROM)
    log(f"  {CHROM} length = {length:,} bp; fetching per-base phyloP (~1 GB float32)")
    arr = bw.values(CHROM, 0, length, numpy=True)
    arr = np.asarray(arr, dtype=np.float32)
    bw.close()
    n_nan = np.isnan(arr).sum()
    log(f"  done; NaN bases = {n_nan:,} ({100*n_nan/length:.2f}%)")
    return arr


def window_means(arr, w):
    """Non-overlapping windows of size w over arr; return NaN-aware mean per window."""
    n = (arr.size // w) * w
    reshaped = arr[:n].reshape(-1, w)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(reshaped, axis=1)
    return mean


def classify(window_means_arr, z=Z_THRESH):
    """Two-tailed Z-score rule on window means: 0=neutral, +1=conserved, -1=accelerated."""
    valid = np.isfinite(window_means_arr)
    mu = window_means_arr[valid].mean()
    sigma = window_means_arr[valid].std(ddof=0)
    labels = np.zeros(window_means_arr.shape, dtype=np.int8)
    labels[window_means_arr > mu + z * sigma] = 1
    labels[window_means_arr < mu - z * sigma] = -1
    labels[~valid] = 0  # NaN → treated as neutral (mostly outside main analysis)
    return labels, mu, sigma


def per_base_labels(window_labels, w, total_len):
    """Broadcast window labels back to per-base resolution (truncate to last full window)."""
    n_full = window_labels.size
    expanded = np.repeat(window_labels, w)
    # Pad to total_len with 0 (neutral) — only affects tail beyond last full window.
    out = np.zeros(total_len, dtype=np.int8)
    out[: expanded.size] = expanded
    return out


def lag1_corr(x):
    """Lag-1 autocorrelation, ignoring NaN by masking."""
    a, b = x[:-1], x[1:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m], b[m]
    return float(np.corrcoef(a, b)[0, 1])


def cohen_kappa(y_true, y_pred):
    """Three-class Cohen's kappa."""
    classes = (-1, 0, 1)
    n = len(y_true)
    po = np.mean(y_true == y_pred)
    pe = 0.0
    for c in classes:
        pe += (np.mean(y_true == c) * np.mean(y_pred == c))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def analysis_1_and_2(arr):
    """Return:
        smoothing_rows: list of dicts for Analysis 1 + bin-fraction columns
        stability_rows: list of dicts for Analysis 2 agreement-vs-100bp
        labels_at_W:    dict {W: per-base label array}
    """
    smoothing_rows = []
    labels_at_W = {}
    for w in WINDOW_SIZES:
        log(f"  W={w}: window means + classification")
        wm = window_means(arr, w)
        finite = wm[np.isfinite(wm)]
        sd = float(finite.std(ddof=0))
        # Median absolute adjacent-window change
        adj = np.abs(np.diff(wm))
        mad_adj = float(np.nanmedian(adj))
        # Lag-1 autocorrelation
        ac1 = lag1_corr(wm)
        labels, mu, sigma = classify(wm)
        pct_con = 100.0 * (labels == 1).mean()
        pct_neu = 100.0 * (labels == 0).mean()
        pct_acc = 100.0 * (labels == -1).mean()
        smoothing_rows.append({
            "window_size": w,
            "n_windows": int(wm.size),
            "mean_phyloP": round(float(np.nanmean(wm)), 4),
            "sd_window_mean": round(sd, 4),
            "median_abs_adj_change": round(mad_adj, 4),
            "lag1_autocorr": round(ac1, 4),
            "global_mu": round(float(mu), 4),
            "global_sigma": round(float(sigma), 4),
            "pct_conserved": round(pct_con, 3),
            "pct_neutral": round(pct_neu, 3),
            "pct_accelerated": round(pct_acc, 3),
        })
        labels_at_W[w] = per_base_labels(labels, w, arr.size)

    # Analysis 2 agreement vs 100bp at base level
    ref = labels_at_W[REFERENCE_W]
    stability_rows = []
    for w in WINDOW_SIZES:
        if w == REFERENCE_W:
            continue
        cmp = labels_at_W[w]
        agree = float((cmp == ref).mean()) * 100
        kappa = cohen_kappa(ref, cmp)
        stability_rows.append({
            "window_size": w,
            "vs": REFERENCE_W,
            "base_level_agreement_pct": round(agree, 3),
            "cohen_kappa": round(float(kappa), 4),
        })
    return smoothing_rows, stability_rows, labels_at_W


def analysis_3(arr, labels_100):
    """Within-100bp-bin heterogeneity, grouped by assigned category."""
    w = REFERENCE_W
    n = (arr.size // w) * w
    bins = arr[:n].reshape(-1, w)               # (n_bins, 100)
    labels_per_bin = labels_100[:n][::w]         # one label per bin (taken from first base)
    # safer: use the window-mean-based labels
    wm_100 = window_means(arr, w)
    labels_per_bin, _, _ = classify(wm_100)
    rows = []
    for cat, name in [(1, "Conserved"), (0, "Neutral"), (-1, "Accelerated")]:
        sel = labels_per_bin == cat
        if not sel.any():
            continue
        sub = bins[sel]                          # (n_cat_bins, 100)
        # per-bin SD (ignoring NaN)
        with np.errstate(invalid="ignore"):
            per_bin_sd = np.nanstd(sub, axis=1, ddof=0)
        # per-bin fraction positive / negative bases
        pos_count = np.nansum(sub > 0, axis=1)
        neg_count = np.nansum(sub < 0, axis=1)
        valid_count = np.sum(np.isfinite(sub), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_pos = pos_count / valid_count
            frac_neg = neg_count / valid_count
        mixed_sign = (pos_count > 0) & (neg_count > 0)
        rows.append({
            "category": name,
            "n_bins": int(sel.sum()),
            "mean_within_bin_SD": round(float(np.nanmean(per_bin_sd)), 4),
            "median_within_bin_SD": round(float(np.nanmedian(per_bin_sd)), 4),
            "pct_mixed_sign_bins": round(100.0 * mixed_sign.mean(), 3),
            "mean_frac_positive_bases": round(float(np.nanmean(frac_pos)), 4),
            "mean_frac_negative_bases": round(float(np.nanmean(frac_neg)), 4),
        })
    return rows


def analysis_4():
    """Analytic boundary-crossing & full-containment probabilities."""
    rows = []
    for B in [10, 25, 50, 100, 200, 500]:
        for L in [6, 8, 12]:
            if L <= B:
                p_cross = (L - 1) / B
                p_contain = (B - L + 1) / B
            else:
                p_cross = 1.0
                p_contain = 0.0
            rows.append({
                "bin_size_bp": B,
                "motif_length_bp": L,
                "p_cross_boundary": round(p_cross, 4),
                "p_fully_contained": round(p_contain, 4),
            })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"  wrote {path} ({len(rows)} rows)")


def main():
    log("Loading per-base phyloP …")
    arr = load_chr_phylop()

    log("=== Analyses 1 & 2: window-size smoothing + category stability ===")
    smoothing_rows, stability_rows, labels_at_W = analysis_1_and_2(arr)

    log("=== Analysis 3: within-100bp-bin heterogeneity ===")
    het_rows = analysis_3(arr, labels_at_W[REFERENCE_W])

    log("=== Analysis 4: analytic boundary-crossing probabilities ===")
    bdry_rows = analysis_4()

    write_csv(OUT / "phylop_window_smoothing_chr1.csv", smoothing_rows)
    write_csv(OUT / "phylop_category_stability_chr1.csv", stability_rows)
    write_csv(OUT / "phylop_within_bin_heterogeneity_chr1.csv", het_rows)
    write_csv(OUT / "phylop_boundary_cross_prob.csv", bdry_rows)

    log("")
    log("=== Smoothing summary ===")
    for r in smoothing_rows:
        log(f"  W={r['window_size']:>4} | SD={r['sd_window_mean']:.4f} | "
            f"med|Δadj|={r['median_abs_adj_change']:.4f} | "
            f"AC1={r['lag1_autocorr']:.4f} | "
            f"%con/neu/acc={r['pct_conserved']:.2f}/{r['pct_neutral']:.2f}/{r['pct_accelerated']:.2f}")
    log("")
    log("=== Stability vs 100bp ===")
    for r in stability_rows:
        log(f"  W={r['window_size']:>4} vs {REFERENCE_W} | "
            f"agree={r['base_level_agreement_pct']:.2f}% | κ={r['cohen_kappa']:.4f}")
    log("")
    log("=== Within-bin heterogeneity (W=100) ===")
    for r in het_rows:
        log(f"  {r['category']:<11} | n_bins={r['n_bins']:>8} | "
            f"mean SD={r['mean_within_bin_SD']:.4f} | %mixed={r['pct_mixed_sign_bins']:.2f}% | "
            f"frac+={r['mean_frac_positive_bases']:.3f} frac-={r['mean_frac_negative_bases']:.3f}")
    log("")
    log("=== Boundary-crossing probabilities (analytic) ===")
    for r in bdry_rows:
        log(f"  B={r['bin_size_bp']:>4} L={r['motif_length_bp']:>3} | "
            f"P(cross)={r['p_cross_boundary']:.4f} | "
            f"P(contained)={r['p_fully_contained']:.4f}")
    log("Done.")


if __name__ == "__main__":
    main()
