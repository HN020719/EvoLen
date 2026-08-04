#!/usr/bin/env bash
set -euo pipefail

# Unified benchmark runner for EvoLen supervised fine-tuning.
# This script is normally called by sft_evolen.py.
#
# Usage:
# bash run_benchmark.sh \
#   data_path \
#   output_path \
#   project_name \
#   model_path \
#   model_name \
#   scale \
#   gpu_id \
#   use_wandb
#
# Args:
#   1) data_path     Final task directory containing train/dev/test CSV files
#   2) output_path   Root directory for fine-tuning outputs
#   3) project_name  W&B project name
#   4) model_path    Hugging Face model ID or local checkpoint path
#   5) model_name    Short model label used in output names
#   6) scale         100k or 200k
#   7) gpu_id        Optional; default: 0
#   8) use_wandb     Optional; default: False

# Load conda and activate the fine-tuning environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bpe

# Locate this script and the repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Read positional arguments supplied by sft_evolen.py.
data_path=${1:?"Missing data_path"}
output_path=${2:?"Missing output_path"}
project_name=${3:?"Missing project_name"}
MODEL=${4:?"Missing model_path"}
MODEL_NAME=${5:?"Missing model_name"}
scale=${6:?"Missing scale"}
gpu_id=${7:-0}
use_wandb=${8:-False}

# Validate the pretraining scale.
case "${scale}" in
    100k|200k)
        ;;
    *)
        echo "Invalid scale: ${scale}. Expected 100k or 200k." >&2
        exit 1
        ;;
esac

# Restrict the process to the selected GPU.
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# Values supplied by sft_evolen.py through environment variables.
BENCHMARK=${BENCHMARK:?"Missing BENCHMARK"}
TASK=${TASK:?"Missing TASK"}
TOKENIZER=${TOKENIZER:?"Missing TOKENIZER"}

best_lr=${LEARNING_RATE:?"Missing LEARNING_RATE"}
best_wd=${WEIGHT_DECAY:?"Missing WEIGHT_DECAY"}
best_wr=${WARMUP_RATIO:?"Missing WARMUP_RATIO"}
best_ws=${WARMUP_STEPS:?"Missing WARMUP_STEPS"}
best_ep=${NUM_TRAIN_EPOCHS:?"Missing NUM_TRAIN_EPOCHS"}
best_seed=${SEED:?"Missing SEED"}

scheduler_type=${LR_SCHEDULER_TYPE:?"Missing LR_SCHEDULER_TYPE"}
train_batch_size=${TRAIN_BATCH_SIZE:?"Missing TRAIN_BATCH_SIZE"}
eval_batch_size=${EVAL_BATCH_SIZE:?"Missing EVAL_BATCH_SIZE"}
gradient_accumulation=${GRADIENT_ACCUMULATION_STEPS:?"Missing GRADIENT_ACCUMULATION_STEPS"}
model_max_length=${MODEL_MAX_LENGTH:?"Missing MODEL_MAX_LENGTH"}

fp16=${FP16:?"Missing FP16"}
evaluation_strategy=${EVALUATION_STRATEGY:?"Missing EVALUATION_STRATEGY"}
save_strategy=${SAVE_STRATEGY:?"Missing SAVE_STRATEGY"}
save_total_limit=${SAVE_TOTAL_LIMIT:?"Missing SAVE_TOTAL_LIMIT"}
load_best_model=${LOAD_BEST_MODEL_AT_END:?"Missing LOAD_BEST_MODEL_AT_END"}
best_model_metric=${METRIC_FOR_BEST_MODEL:?"Missing METRIC_FOR_BEST_MODEL"}
greater_is_better=${GREATER_IS_BETTER:?"Missing GREATER_IS_BETTER"}

# Select the benchmark-specific Python training file.
case "${BENCHMARK}" in
    NT)
        TRAIN_SCRIPT="${REPO_ROOT}/train_scripts/NT_train.py"
        ;;
    GUE)
        TRAIN_SCRIPT="${REPO_ROOT}/train_scripts/GUE_train.py"
        ;;
    GBM)
        TRAIN_SCRIPT="${REPO_ROOT}/train_scripts/GBM_train.py"
        ;;
    ATAC)
        TRAIN_SCRIPT="${REPO_ROOT}/train_scripts/ATAC_train.py"
        ;;
    SCREEN)
        TRAIN_SCRIPT="${REPO_ROOT}/train_scripts/screen_train.py"
        ;;
    *)
        echo "Unsupported benchmark: ${BENCHMARK}" >&2
        exit 1
        ;;
esac

# Normalize common Boolean representations to True or False.
normalize_bool() {
    case "${1,,}" in
        true|1|yes)
            printf "True"
            ;;
        false|0|no)
            printf "False"
            ;;
        *)
            echo "Invalid Boolean value: ${1}" >&2
            return 1
            ;;
    esac
}

use_wandb="$(normalize_bool "${use_wandb}")"
fp16="$(normalize_bool "${fp16}")"
load_best_model="$(normalize_bool "${load_best_model}")"
greater_is_better="$(normalize_bool "${greater_is_better}")"

# Resolve a relative data path when this script is run manually.
if [[ ! -d "${data_path}" && -d "${SCRIPT_DIR}/${data_path}" ]]; then
    data_path="${SCRIPT_DIR}/${data_path}"
elif [[ ! -d "${data_path}" && -d "${REPO_ROOT}/${data_path}" ]]; then
    data_path="${REPO_ROOT}/${data_path}"
fi

# Validate required paths and dataset files.
if [[ ! -d "${data_path}" ]]; then
    echo "data_path does not exist: ${data_path}" >&2
    exit 1
fi

for split_file in train.csv dev.csv test.csv; do
    if [[ ! -f "${data_path}/${split_file}" ]]; then
        echo "Dataset file does not exist: ${data_path}/${split_file}" >&2
        exit 1
    fi
done

if [[ ! -f "${TOKENIZER}" ]]; then
    echo "TOKENIZER does not exist: ${TOKENIZER}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "TRAIN_SCRIPT does not exist: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# Replace slashes in GUE task names such as EMP/H3.
task_slug="${TASK//\//_}"

# Create descriptive names for the run and output directory.
hp_tag="lr${best_lr}_wd${best_wd}_wr${best_wr}_ep${best_ep}_seed${best_seed}"
run_name="${BENCHMARK}_${MODEL_NAME}_${task_slug}_${hp_tag}"

# TASK may contain a slash. For example, EMP/H3 becomes nested
# output folders while the run name uses EMP_H3.
run_output_dir="${output_path}/${BENCHMARK}/${TASK}/${MODEL_NAME}/${hp_tag}"

# Skip the run when its final evaluation results already exist.
result_json="${run_output_dir}/results/${run_name}/eval_results.json"

if [[ -f "${result_json}" ]]; then
    echo "[SKIP] ${run_name}"
    exit 0
fi

mkdir -p "${run_output_dir}"

echo "===== ${BENCHMARK} TASK: ${TASK} ====="
echo "[RUN ] ${run_name}"
echo "MODEL=${MODEL}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SCALE=${scale}"
echo "TOKENIZER=${TOKENIZER}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "DATA=${data_path}"
echo "OUTPUT=${run_output_dir}"
echo "GPU=${CUDA_VISIBLE_DEVICES}"
echo "USE_WANDB=${use_wandb}"
echo "LEARNING_RATE=${best_lr}"
echo "WEIGHT_DECAY=${best_wd}"
echo "WARMUP_RATIO=${best_wr}"
echo "WARMUP_STEPS=${best_ws}"
echo "EPOCHS=${best_ep}"
echo "SCHEDULER=${scheduler_type}"
echo "TRAIN_BATCH_SIZE=${train_batch_size}"
echo "EVAL_BATCH_SIZE=${eval_batch_size}"
echo "GRADIENT_ACCUMULATION=${gradient_accumulation}"
echo "MODEL_MAX_LENGTH=${model_max_length}"
echo "FP16=${fp16}"
echo "METRIC_FOR_BEST_MODEL=${best_model_metric}"

# Build the shared training command.
cmd=(
    python "${TRAIN_SCRIPT}"
    --model_name_or_path "${MODEL}"
    --tokenizer_path "${TOKENIZER}"
    --trust_remote_code True
    --data_path "${data_path}"
    --kmer -1
    --run_name "${run_name}"
    --model_max_length "${model_max_length}"
    --per_device_train_batch_size "${train_batch_size}"
    --per_device_eval_batch_size "${eval_batch_size}"
    --gradient_accumulation_steps "${gradient_accumulation}"
    --learning_rate "${best_lr}"
    --weight_decay "${best_wd}"
    --num_train_epochs "${best_ep}"
    --lr_scheduler_type "${scheduler_type}"
    --warmup_steps "${best_ws}"
    --warmup_ratio "${best_wr}"
)

# --fp16 is a flag, so include it only when enabled.
if [[ "${fp16}" == "True" ]]; then
    cmd+=(--fp16)
fi

cmd+=(
    --output_dir "${run_output_dir}"
    --evaluation_strategy "${evaluation_strategy}"
    --save_strategy "${save_strategy}"
    --load_best_model_at_end "${load_best_model}"
    --metric_for_best_model "${best_model_metric}"
    --greater_is_better "${greater_is_better}"
    --save_total_limit "${save_total_limit}"
    --logging_steps 100
    --overwrite_output_dir True
    --log_level info
    --seed "${best_seed}"
    --find_unused_parameters False
    --use_wandb "${use_wandb}"
    --wandb_project "${project_name}"
)

# Print the final command in a copyable format.
printf "COMMAND="
printf "%q " "${cmd[@]}"
printf "\n"

"${cmd[@]}"
