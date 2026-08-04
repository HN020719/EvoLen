#!/usr/bin/env python3

# Unified CLI for selecting and running EvoLen SFT benchmark tasks.

import argparse
import csv
import os
import shlex
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "assets" / "evolen_unified_manifest.csv"
RUN_SCRIPT = SCRIPT_DIR / "run_benchmark.sh"

MANIFEST_BENCHMARKS = {
    "NT": "NT",
    "GUE": "GUE",
    "GBM": "GBM",
    "ATAC": "multi-ATAC",
    "SCREEN": "multi-SCREEN",
}

# Map environment variable names to manifest column names.
HP_ENV_COLUMNS = {
    "LEARNING_RATE": "learning_rate",
    "WEIGHT_DECAY": "weight_decay",
    "WARMUP_RATIO": "warmup_ratio",
    "WARMUP_STEPS": "warmup_steps",
    "NUM_TRAIN_EPOCHS": "num_train_epochs",
    "LR_SCHEDULER_TYPE": "lr_scheduler_type",
    "SEED": "seed",
    "TRAIN_BATCH_SIZE": "per_device_train_batch_size",
    "EVAL_BATCH_SIZE": "per_device_eval_batch_size",
    "GRADIENT_ACCUMULATION_STEPS": "gradient_accumulation_steps",
    "FP16": "fp16",
    "MODEL_MAX_LENGTH": "model_max_length",
    "EVALUATION_STRATEGY": "evaluation_strategy",
    "SAVE_STRATEGY": "save_strategy",
    "SAVE_TOTAL_LIMIT": "save_total_limit",
    "LOAD_BEST_MODEL_AT_END": "load_best_model_at_end",
    "METRIC_FOR_BEST_MODEL": "metric_for_best_model",
    "GREATER_IS_BETTER": "greater_is_better",
}

LOWERCASE_ENV_VARS = {
    "LR_SCHEDULER_TYPE",
    "EVALUATION_STRATEGY",
    "SAVE_STRATEGY",
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="EvoLen SFT",
        description="Runs supervised fine-tuning for the EvoLen paper.",
    )

    parser.add_argument(
        "-s",
        "--scale",
        type=str.lower,
        choices=["100k", "200k"],
        default="100k",
        help="Pretraining scale. Default: 100k.",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str.lower,
        choices=["evolen", "base"],
        default="evolen",
        help="Model to fine-tune. Default: evolen.",
    )

    parser.add_argument(
        "-b",
        "--benchmark",
        type=str.upper,
        choices=["NT", "GUE", "GBM", "ATAC", "SCREEN"],
        required=True,
        help="Benchmark suite to run.",
    )

    parser.add_argument(
        "-t",
        "--task",
        default=None,
        help=(
            "Specific task. May be omitted when a benchmark "
            "has only one task."
        ),
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Root directory of the downloaded benchmark dataset.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory where fine-tuning outputs will be saved.",
    )

    parser.add_argument(
        "--project-name",
        default="EvoLen-SFT",
        help="W&B project name. Default: EvoLen-SFT.",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use. Default: 0.",
    )

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable W&B. Default: disabled.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without starting training.",
    )

    return parser, parser.parse_args()


def resolve_path(path_text, root):
    path = Path(path_text).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (root / path).resolve()


def main():
    parser, args = parse_args()

    benchmark_data_path = args.data_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not MANIFEST_PATH.is_file():
        parser.error(f"Manifest does not exist: {MANIFEST_PATH}")

    with MANIFEST_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "scale",
            "model",
            "benchmark",
            "task",
            "repo",
            "subfolder",
            "pretrained_model",
            "tokenizer",
            *HP_ENV_COLUMNS.values(),
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            parser.error(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        rows = list(reader)

    manifest_benchmark = MANIFEST_BENCHMARKS[args.benchmark]

    benchmark_rows = [
        row
        for row in rows
        if row["model"].strip().lower() == args.model
        and row["scale"].strip().lower() == args.scale
        and row["benchmark"].strip().lower()
        == manifest_benchmark.lower()
    ]

    if not benchmark_rows:
        parser.error(
            "No manifest rows found for "
            f"{args.model}, {args.scale}, {args.benchmark}."
        )

    available_tasks = sorted(
        {
            row["task"].strip()
            for row in benchmark_rows
        }
    )

    if args.task is None:
        if len(available_tasks) != 1:
            parser.error(
                f"--task is required for {args.benchmark}. "
                "Available tasks: "
                + ", ".join(available_tasks)
            )

        task = available_tasks[0]

    else:
        task_matches = [
            task_name
            for task_name in available_tasks
            if task_name.lower() == args.task.lower()
        ]

        if not task_matches:
            parser.error(
                f"Unknown task '{args.task}'. "
                "Available tasks: "
                + ", ".join(available_tasks)
            )

        task = task_matches[0]

    matching_rows = [
        row
        for row in benchmark_rows
        if row["task"].strip() == task
    ]

    if len(matching_rows) != 1:
        parser.error(
            "Expected exactly one manifest row for "
            f"{args.model}/{args.scale}/{args.benchmark}/{task}, "
            f"but found {len(matching_rows)}."
        )

    config = matching_rows[0]

    # Construct the final task-specific dataset path.
    #
    # NT and GBM store CSV files inside:
    #   <benchmark root>/<task>/split
    #
    # GUE, ATAC, and SCREEN store CSV files inside:
    #   <benchmark root>/<task>
    task_relative_path = Path(task)

    if task_relative_path.is_absolute():
        parser.error(
            f"Manifest task must be a relative path: {task}"
        )

    task_data_path = (
        benchmark_data_path / task_relative_path
    ).resolve()

    if args.benchmark in {"NT", "GBM"}:
        task_data_path = task_data_path / "split"

    if not task_data_path.is_dir():
        parser.error(
            "Task directory does not exist:\n"
            f"  Benchmark root: {benchmark_data_path}\n"
            f"  Benchmark: {args.benchmark}\n"
            f"  Task: {task}\n"
            f"  Resolved directory: {task_data_path}"
        )

    required_dataset_files = [
        task_data_path / "train.csv",
        task_data_path / "dev.csv",
        task_data_path / "test.csv",
    ]

    missing_dataset_files = [
        path
        for path in required_dataset_files
        if not path.is_file()
    ]

    if missing_dataset_files:
        parser.error(
            "Task directory is missing required dataset files:\n  "
            + "\n  ".join(
                str(path)
                for path in missing_dataset_files
            )
        )

    tokenizer_path = resolve_path(
        config["tokenizer"],
        SCRIPT_DIR,
    )

    if not tokenizer_path.is_file():
        parser.error(
            f"Tokenizer file does not exist: {tokenizer_path}"
        )

    pretrained_model = config["pretrained_model"].strip()

    if not pretrained_model:
        parser.error(
            "The selected manifest row has an empty "
            "pretrained_model value."
        )

    run_script = RUN_SCRIPT

    if not run_script.is_file():
        parser.error(
            f"Unified run script does not exist: {run_script}"
        )

    if args.model == "evolen":
        model_name = f"evolen-{args.scale}"
    else:
        model_name = f"base-{args.scale}"

    use_wandb = "True" if args.wandb else "False"

    # Positional arguments expected by run_benchmark.sh:
    #
    # 1) data_path
    # 2) output_path
    # 3) project_name
    # 4) model_path
    # 5) model_name
    # 6) scale
    # 7) gpu_id
    # 8) use_wandb
    command = [
        "bash",
        str(run_script),
        str(task_data_path),
        str(output_path),
        args.project_name,
        pretrained_model,
        model_name,
        args.scale,
        str(args.gpu_id),
        use_wandb,
    ]

    # Pass manifest hyperparameters to run_benchmark.sh through
    # environment variables.
    env = os.environ.copy()

    for env_name, column_name in HP_ENV_COLUMNS.items():
        value = config[column_name].strip()

        if not value:
            parser.error(
                "Manifest value is empty for "
                f"{column_name} in "
                f"{args.benchmark}/{task}."
            )

        if env_name in LOWERCASE_ENV_VARS:
            value = value.lower()

        env[env_name] = value

    env.update(
        {
            "TOKENIZER": str(tokenizer_path),
            "BENCHMARK": args.benchmark,
            "TASK": task,
            "CHECKPOINT_REPO": config["repo"].strip(),
            "CHECKPOINT_SUBFOLDER": config["subfolder"].strip(),
            "PYTHONUNBUFFERED": "1",
        }
    )

    print("===== EvoLen SFT configuration =====")
    print(f"Model:       {args.model}")
    print(f"Model name:  {model_name}")
    print(f"Scale:       {args.scale}")
    print(f"Benchmark:   {args.benchmark}")
    print(f"Task:        {task}")
    print(f"Checkpoint:  {pretrained_model}")
    print(f"Tokenizer:   {tokenizer_path}")
    print(f"Data root:   {benchmark_data_path}")
    print(f"Task data:   {task_data_path}")
    print(f"Output:      {output_path}")
    print(f"GPU:         {args.gpu_id}")
    print(f"W&B:         {args.wandb}")
    print(f"Run script:  {run_script}")
    print("Command:")
    print(shlex.join(command))

    if args.dry_run:
        print("Dry run complete. Training was not started.")
        return

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        command,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
