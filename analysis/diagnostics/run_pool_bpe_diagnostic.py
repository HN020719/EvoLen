#!/usr/bin/env python3
"""Compute compression and entropy diagnostics for the three pool-specific BPEs
(conserved / neutral / accelerated) across vocab sizes {2048, 3072, 4096, 5120}.

Uses hg38 chr1 (N-stripped) as a common test corpus.
"""
import json
import math
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pysam
from tokenizers import Tokenizer

import os as _os
EVOLEN_ROOT = _os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")



def load_tokenizer_compat(path: Path) -> Tokenizer:
    """Load a BPE tokenizer JSON, converting list-of-pair merges to old-format
    (space-joined strings) for older `tokenizers` libraries (<0.19)."""
    try:
        return Tokenizer.from_file(str(path))
    except Exception:
        pass
    with open(path) as f:
        data = json.load(f)
    merges = data.get("model", {}).get("merges")
    if isinstance(merges, list) and merges and isinstance(merges[0], list):
        data["model"]["merges"] = [" ".join(p) for p in merges]
    # Some new-format fields aren't recognized by older lib — drop if needed.
    data["model"].pop("ignore_merges", None)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(data, tmp)
        tmp_path = tmp.name
    return Tokenizer.from_file(tmp_path)

ROOT = Path(f"{EVOLEN_ROOT}")
MERGE_BPE_DIR = Path(f"{EVOLEN_ROOT}/tokenizer_evaluation/merge_bpe")
FASTA = ROOT / "hg38.fa"
OUT_DIR = ROOT / "tokenizer_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZES = [2048, 3072, 4096, 5120]
POOLS = {"Conserved": "con_tokenizer.json",
         "Neutral":   "neu_tokenizer.json",
         "Accelerated":"acc_tokenizer.json"}
CHROM = "chr1"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_corpus():
    log(f"Loading {CHROM} from {FASTA} ...")
    with pysam.FastaFile(str(FASTA)) as fa:
        seq = fa.fetch(CHROM).upper().replace("N", "")
    log(f"  {CHROM} N-stripped length = {len(seq):,} bp")
    return seq


def chunked(seq, size=200_000):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def encode_corpus(tok, seq):
    counts = Counter()
    n_tok = 0
    for chunk in chunked(seq):
        for enc in tok.encode_batch([chunk], add_special_tokens=False):
            counts.update(enc.tokens)
            n_tok += len(enc.tokens)
    return counts, n_tok


def diagnostics(counts, n_tok, n_bp):
    bp_per_tok = n_bp / n_tok
    tok_per_kb = 1000.0 * n_tok / n_bp
    total = sum(counts.values())
    H = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    H_max = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return bp_per_tok, tok_per_kb, H, H_max, len(counts)


def main():
    seq = load_corpus()
    n_bp = len(seq)

    rows = []
    jaccard_rows = []

    for vocab_size in VOCAB_SIZES:
        log(f"=== Vocab size {vocab_size} ===")
        tokenizers = {}
        vocabs = {}
        for pool_name, fname in POOLS.items():
            p = MERGE_BPE_DIR / f"vocab_{vocab_size}" / fname
            if not p.exists():
                log(f"  [missing] {p}")
                continue
            tokenizers[pool_name] = load_tokenizer_compat(p)
            vocabs[pool_name] = set(tokenizers[pool_name].get_vocab().keys())
            log(f"  loaded {pool_name}: {tokenizers[pool_name].get_vocab_size()} tokens")

        for pool_name, tok in tokenizers.items():
            log(f"  encoding with {pool_name} ...")
            counts, n_tok = encode_corpus(tok, seq)
            bp_t, t_kb, H, H_max, vocab_used = diagnostics(counts, n_tok, n_bp)
            rows.append({
                "vocab_size": vocab_size,
                "pool": pool_name,
                "vocab_size_loaded": tok.get_vocab_size(),
                "vocab_size_used": vocab_used,
                "bp_per_token": round(bp_t, 4),
                "tokens_per_kb": round(t_kb, 4),
                "shannon_H_bits": round(H, 4),
                "H_normalized": round(H / H_max, 4),
                "total_tokens": n_tok,
            })
            log(f"    bp/tok={bp_t:.3f}  tokens/kb={t_kb:.2f}  H={H:.4f}  H_norm={H/H_max:.4f}")

        for a in vocabs:
            for b in vocabs:
                if a >= b:
                    continue
                inter = len(vocabs[a] & vocabs[b])
                union = len(vocabs[a] | vocabs[b])
                jaccard_rows.append({
                    "vocab_size": vocab_size,
                    "pool_a": a,
                    "pool_b": b,
                    "intersection": inter,
                    "union": union,
                    "jaccard": round(inter / union, 4),
                })
                log(f"  Jaccard {a} ∩ {b} = {inter/union:.4f}")

    # Write outputs
    import csv
    comp_out = OUT_DIR / "pool_bpe_compression_entropy.csv"
    with comp_out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"Saved compression/entropy table -> {comp_out}")

    jac_out = OUT_DIR / "pool_bpe_jaccard.csv"
    with jac_out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(jaccard_rows[0].keys()))
        w.writeheader()
        w.writerows(jaccard_rows)
    log(f"Saved Jaccard table             -> {jac_out}")

    # Pretty-print summary
    log("")
    log("=== Compression / Entropy summary ===")
    log(f"{'vocab':>6} {'pool':<13} {'bp/tok':>8} {'tokens/kb':>11} {'H (bits)':>10} {'H_norm':>8}")
    for r in rows:
        log(f"{r['vocab_size']:>6} {r['pool']:<13} {r['bp_per_token']:>8.3f} "
            f"{r['tokens_per_kb']:>11.2f} {r['shannon_H_bits']:>10.4f} {r['H_normalized']:>8.4f}")
    log("")
    log("=== Jaccard summary ===")
    log(f"{'vocab':>6} {'pair':<28} {'jaccard':>8}")
    for r in jaccard_rows:
        log(f"{r['vocab_size']:>6} {r['pool_a']:<13}-{r['pool_b']:<13} {r['jaccard']:>8.4f}")


if __name__ == "__main__":
    main()
