import argparse
import glob
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train/val/test files from cCRE *_positive.fa FASTA files "
            "for multiclass or binary fine-tuning."
        )
    )
    parser.add_argument(
        "--pattern",
        default="*_positive.fa",
        help="Glob pattern for input FASTA files (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="finetune_ready",
        help="Directory to write JSONL outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling/splitting (default: %(default)s).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of examples used for the training split (default: %(default)s).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of examples used for the validation split (default: %(default)s).",
    )
    parser.add_argument(
        "--binary-per-class",
        action="store_true",
        help=(
            "If set, build separate train/val/test files for each class using "
            "both *_positive.fa and *_negative.fa files."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format. For binary-per-class, CSV will mirror text/label columns.",
    )
    return parser.parse_args()


def read_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    header = None
    seq_chunks: List[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def derive_label(fasta_path: Path) -> str:
    name = fasta_path.name
    for suffix in ("_positive.fa", "_negative.fa"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if stem.startswith("chr") and "_" in stem:
                prefix, remainder = stem.split("_", 1)
                if prefix[3:] in {
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                    "16",
                    "17",
                    "18",
                    "19",
                    "20",
                    "21",
                    "22",
                    "X",
                    "Y",
                    "M",
                }:
                    return remainder
            return stem

    raise ValueError(f"Unexpected FASTA name: {name}")


def build_entry(sequence: str, label: str, header: str) -> Dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a DNA cCRE classifier. Given a human genomic sequence, "
                    "respond with exactly one label from: CA, CA-CTCF, CA-H3K4me3, "
                    "CA-TF, dELS, pELS, PLS, TF."
                ),
            },
            {
                "role": "user",
                "content": f"Sequence: {sequence}\nID: {header}",
            },
            {"role": "assistant", "content": label},
        ]
    }


def build_binary_entry(
    sequence: str, class_label: str, header: str, is_positive: bool
) -> Dict:
    verdict = "positive" if is_positive else "negative"
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a DNA cCRE classifier for the {class_label} cell type. "
                    "Given a human genomic sequence, respond with exactly one label: "
                    "positive or negative."
                ),
            },
            {
                "role": "user",
                "content": f"Sequence: {sequence}\nID: {header}\nTarget: {class_label}",
            },
            {"role": "assistant", "content": verdict},
        ]
    }


def split_data(
    items: List[Any], train_ratio: float, val_ratio: float
) -> Tuple[List[Any], List[Any], List[Any]]:
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train = items[:train_end]
    val = items[train_end:val_end]
    test = items[val_end:]
    return train, val, test


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("text,label\n")
        for text, label in rows:
            handle.write(f"{text},{label}\n")


def build_binary_per_class(
    args: argparse.Namespace, rng: random.Random
) -> None:
    positive_files = sorted(glob.glob(args.pattern))
    if not positive_files:
        raise FileNotFoundError(f"No FASTA files match pattern {args.pattern}")

    for positive_path_str in positive_files:
        positive_path = Path(positive_path_str)
        class_label = derive_label(positive_path)
        negative_path = positive_path.with_name(
            positive_path.name.replace("_positive.fa", "_negative.fa")
        )
        if not negative_path.exists():
            raise FileNotFoundError(
                f"Missing negative FASTA for {class_label}: {negative_path}"
            )

        records: List[Tuple[str, str, bool]] = []
        for header, seq in read_fasta(positive_path):
            records.append((header, seq, True))
        for header, seq in read_fasta(negative_path):
            records.append((header, seq, False))

        rng.shuffle(records)
        train, val, test = split_data(records, args.train_ratio, args.val_ratio)

        output_dir = Path(args.output_dir) / class_label

        if args.format == "jsonl":
            write_jsonl(
                output_dir / "train.jsonl",
                [build_binary_entry(seq, class_label, header, is_pos) for header, seq, is_pos in train],
            )
            write_jsonl(
                output_dir / "val.jsonl",
                [build_binary_entry(seq, class_label, header, is_pos) for header, seq, is_pos in val],
            )
            write_jsonl(
                output_dir / "test.jsonl",
                [build_binary_entry(seq, class_label, header, is_pos) for header, seq, is_pos in test],
            )
        else:
            write_csv(
                output_dir / "train.csv",
                [(seq, "positive" if is_pos else "negative") for _, seq, is_pos in train],
            )
            write_csv(
                output_dir / "dev.csv",
                [(seq, "positive" if is_pos else "negative") for _, seq, is_pos in val],
            )
            write_csv(
                output_dir / "test.csv",
                [(seq, "positive" if is_pos else "negative") for _, seq, is_pos in test],
            )

        split_names = ("train", "val" if args.format == "jsonl" else "dev", "test")
        split_counts = (len(train), len(val), len(test))
        print(
            f"{class_label}: wrote {split_counts[0]} {split_names[0]}, "
            f"{split_counts[1]} {split_names[1]}, {split_counts[2]} {split_names[2]} "
            f"examples to {output_dir}"
        )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.binary_per_class:
        build_binary_per_class(args, rng)
        return

    all_entries: List[Dict] = []
    all_rows: List[Tuple[str, str]] = []
    for fasta in sorted(glob.glob(args.pattern)):
        fasta_path = Path(fasta)
        label = derive_label(fasta_path)
        for header, seq in read_fasta(fasta_path):
            if args.format == "jsonl":
                all_entries.append(build_entry(seq, label, header))
            else:
                all_rows.append((seq, label))

    output_dir = Path(args.output_dir)
    if args.format == "jsonl":
        rng.shuffle(all_entries)
        train, val, test = split_data(all_entries, args.train_ratio, args.val_ratio)
        write_jsonl(output_dir / "train.jsonl", train)
        write_jsonl(output_dir / "val.jsonl", val)
        write_jsonl(output_dir / "test.jsonl", test)
        print(
            f"Wrote {len(train)} train, {len(val)} val, {len(test)} test examples "
            f"to {output_dir}"
        )
    else:
        rng.shuffle(all_rows)
        train, dev, test = split_data(all_rows, args.train_ratio, args.val_ratio)
        write_csv(output_dir / "train.csv", train)
        write_csv(output_dir / "dev.csv", dev)
        write_csv(output_dir / "test.csv", test)
        print(
            f"Wrote {len(train)} train, {len(dev)} dev, {len(test)} test examples "
            f"to {output_dir}"
        )


if __name__ == "__main__":
    main()
