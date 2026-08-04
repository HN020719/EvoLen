#!/usr/bin/env bash
# MLM pretraining to 100k steps (the checkpoints released as evolen-100k / base-100k).
#
# Usage:
#   ./04.pretrain.sh [merge_bpe_5120|baseline_bpe_5120]
#
# Environment:
#   EVOLEN_DATA_DIR   directory holding *_allchr_all_tokenized_{train,val}_chrOnly.tsv
#                     (download from EvoLenTokenizer/pretrain, or build with tok_split_full.py)
#   EVOLEN_OUT_DIR    where checkpoints are written
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RUN=${1:-merge_bpe_5120}
DATA_DIR=${EVOLEN_DATA_DIR:-$HOME/evolen_data/output_tokens}
OUT_DIR=${EVOLEN_OUT_DIR:-$HOME/evolen_models/${RUN}}
CACHE_DIR=${DATA_DIR}/cache

case "${RUN}" in
  merge_bpe_5120)    TOKENIZER=../assets/tokenizers/merge_tokenizer_unigram_len2.json ;;
  baseline_bpe_5120) TOKENIZER=../assets/tokenizers/5120_tokenizer.json ;;
  *) echo "unknown run: ${RUN} (expected merge_bpe_5120 or baseline_bpe_5120)" >&2; exit 1 ;;
esac

mkdir -p "${OUT_DIR}"

torchrun --nproc_per_node="${NPROC:-8}" run_mlm.py \
    --output_dir "${OUT_DIR}" \
    --model_type bert \
    --tokenizer_name "${TOKENIZER}" \
    --config_name configs/config_5120.json \
    --project_name "pretrain_${RUN}" \
    --do_train True \
    --model_max_length 512 \
    --max_seq_length 512 \
    --line_by_line True \
    --pad_to_max_length True \
    --train_file "${DATA_DIR}/${RUN}_allchr_all_tokenized_train_chrOnly.tsv" \
    --validation_file "${DATA_DIR}/${RUN}_allchr_all_tokenized_val_chrOnly.tsv" \
    --cache_dir "${CACHE_DIR}" \
    --use_fast_tokenizer True \
    --do_eval True \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 96 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --max_steps 100000 \
    --logging_steps 1000 \
    --learning_rate 4e-5 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --mlm_probability 0.15 \
    --warmup_steps 10000 \
    --seed 42 \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir True
