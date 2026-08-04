#!/usr/bin/env python3
"""Run motif coverage eval for ablation tokenizers + references (baseline_bpe_5120, merge_uni_len2_5120)."""

import sys
from pathlib import Path

# Monkey-patch the tokenizer paths before importing
import motif_coverage_eval as mce

import os as _os
EVOLEN_ROOT = _os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")


mce.DEFAULT_TOKENIZER_PATHS = {
    "baseline_bpe_5120":     "tokenizer_evaluation/baseline_bpe/vocab_5120/5120_tokenizer.json",
    "merge_uni_len2_5120":   "tokenizer_evaluation/merge_bpe/vocab_5120/merge_tokenizer_unigram_len2.json",
    "ablation_no_partition":  "tokenizer_evaluation/ablation/vocab_5120/ablation_no_partition.json",
    "ablation_no_priority":   "tokenizer_evaluation/ablation/vocab_5120/ablation_no_priority.json",
    "ablation_no_length":     "tokenizer_evaluation/ablation/vocab_5120/ablation_no_length.json",
}

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--motif-file", f"{EVOLEN_ROOT}/motifs.txt",
        "--output-dir", f"{EVOLEN_ROOT}/tokenizer_evaluation/motif_eval_outputs_ablation",
    ]
    mce.main()
