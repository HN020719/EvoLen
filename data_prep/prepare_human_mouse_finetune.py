import argparse
import csv
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os as _os
EVOLEN_ROOT = _os.environ.get("EVOLEN_ROOT", "/home/n5huang/dna_token")



@dataclass(frozen=True)
class PeakRecord:
    chrom: str
    start: int
    end: int
    labels: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/dev/test CSVs from human/mouse peak TSVs."
    )
    parser.add_argument(
        "--human-tsv",
        type=Path,
        default=Path(f"{EVOLEN_ROOT}/Finetune-species/chr1_Supplementary_Table_4_Human_ATAC_peaks.tsv"),
        help="Human peak TSV with hg38_coord + label column (default: %(default)s).",
    )
    parser.add_argument(
        "--mouse-tsv",
        type=Path,
        default=Path(f"{EVOLEN_ROOT}/Finetune-species/chr1_Supplementary_Table_7_Mouse_ATAC_peaks.tsv"),
        help="Mouse peak TSV with coord + label column (default: %(default)s).",
    )
    parser.add_argument(
        "--label-col",
        default="celltype",
        help="Column name for labels in TSVs (default: %(default)s).",
    )
    parser.add_argument(
        "--human-fasta",
        type=Path,
        default=Path(f"{EVOLEN_ROOT}/hg38.fa"),
        help="Human genome FASTA (default: %(default)s).",
    )
    parser.add_argument(
        "--mouse-fasta",
        type=Path,
        help="Mouse genome FASTA (mm10) for evaluation data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"{EVOLEN_ROOT}/finetune_data/human_mouse_celltype"),
        help="Where to write train/dev/test CSVs (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: %(default)s)."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction for train split from human data (default: %(default)s).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction for dev/val split from human data (default: %(default)s).",
    )
    parser.add_argument(
        "--multi-label",
        choices=("first", "drop", "explode"),
        default="explode",
        help="How to handle multi-celltype rows (default: %(default)s).",
    )
    parser.add_argument(
        "--chrom",
        default=None,
        help="Restrict to a single chromosome (e.g., chr1).",
    )
    parser.add_argument(
        "--use-bedtools",
        action="store_true",
        default=False,
        help="Use bedtools getfasta instead of .fai indexing (default: False).",
    )
    parser.add_argument(
        "--skip-mouse-test",
        action="store_true",
        default=True,
        help="Skip building mouse test split (default: False).",
    )
    parser.add_argument(
        "--no-skip-mouse-test",
        dest="skip_mouse_test",
        action="store_false",
        help="Build mouse test split.",
    )
    parser.add_argument(
        "--match-mouse-test-to-dev",
        action="store_true",
        default=False,
        help="Subsample mouse test to match human dev size (default: False).",
    )
    parser.add_argument(
        "--shared-labels-only",
        action="store_true",
        default=True,
        help="Keep only labels shared between human and mouse (default: True).",
    )
    parser.add_argument(
        "--no-shared-labels-only",
        dest="shared_labels_only",
        action="store_false",
        help="Keep all labels without enforcing intersection.",
    )
    return parser.parse_args()


def parse_coord(raw: str) -> Tuple[str, int, int]:
    chrom, start, end = raw.split("-")
    return chrom, int(start), int(end)


def split_celltypes(raw: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def iter_peak_records(
    path: Path, coord_col: str, label_col: str, chrom_filter: Optional[str]
) -> Iterable[PeakRecord]:
    with path.open() as handle:
        header = handle.readline().strip().split("\t")
        coord_idx = header.index(coord_col)
        label_idx = header.index(label_col)
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            chrom, start, end = parse_coord(parts[coord_idx])
            if chrom_filter and chrom != chrom_filter:
                continue
            labels = split_celltypes(parts[label_idx])
            if not labels:
                continue
            yield PeakRecord(chrom=chrom, start=start, end=end, labels=labels)


def expand_labels(record: PeakRecord, policy: str) -> Iterable[PeakRecord]:
    if policy == "drop":
        if len(record.labels) == 1:
            yield record
        return
    if policy == "first":
        yield PeakRecord(record.chrom, record.start, record.end, (record.labels[0],))
        return
    for label in record.labels:
        yield PeakRecord(record.chrom, record.start, record.end, (label,))


class FastaIndex:
    def __init__(self, fasta_path: Path):
        self.fasta_path = fasta_path
        self.index: Dict[str, Tuple[int, int, int, int]] = {}
        self._load_index()

    def _load_index(self) -> None:
        fai_path = self.fasta_path.with_suffix(self.fasta_path.suffix + ".fai")
        if not fai_path.exists():
            raise FileNotFoundError(f"Missing FASTA index: {fai_path}")
        with fai_path.open() as handle:
            for line in handle:
                name, length, offset, line_blen, line_len = line.strip().split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(line_blen), int(line_len))

    def fetch(self, chrom: str, start: int, end: int) -> str:
        if chrom not in self.index:
            raise KeyError(f"Chrom not found in FASTA index: {chrom}")
        length, offset, line_blen, line_len = self.index[chrom]
        start0 = start - 1
        end0 = end
        if start0 < 0 or end0 > length or start0 >= end0:
            raise ValueError(f"Invalid range for {chrom}:{start}-{end}")
        line_start = start0 // line_blen
        line_end = (end0 - 1) // line_blen
        line_offset = start0 % line_blen
        byte_offset = offset + line_start * line_len + line_offset
        byte_count = (line_end - line_start) * line_len + (end0 - 1) % line_blen + 1 - line_offset
        with self.fasta_path.open("rb") as handle:
            handle.seek(byte_offset)
            raw = handle.read(byte_count).decode("ascii")
        return raw.replace("\n", "").replace("\r", "")


def bedtools_getfasta(
    fasta_path: Path, records: Sequence[PeakRecord]
) -> List[Tuple[PeakRecord, str]]:
    if shutil.which("bedtools") is None:
        raise FileNotFoundError("bedtools not found in PATH.")
    with tempfile.TemporaryDirectory() as tmpdir:
        bed_path = Path(tmpdir) / "peaks.bed"
        out_path = Path(tmpdir) / "peaks.fa"
        with bed_path.open("w") as handle:
            for record in records:
                # bedtools expects 0-based, half-open coordinates.
                handle.write(
                    f"{record.chrom}\t{record.start - 1}\t{record.end}\t{record.labels[0]}\n"
                )
        subprocess.run(
            ["bedtools", "getfasta", "-fi", str(fasta_path), "-bed", str(bed_path), "-fo", str(out_path)],
            check=True,
        )
        seqs: List[Tuple[PeakRecord, str]] = []
        with out_path.open() as handle:
            current = None
            chunks: List[str] = []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current is not None:
                        seqs.append((current, "".join(chunks)))
                    current = records[len(seqs)]
                    chunks = []
                else:
                    chunks.append(line)
            if current is not None:
                seqs.append((current, "".join(chunks)))
    return seqs


def load_rows(
    records: Sequence[PeakRecord],
    fasta: FastaIndex,
    label_filter: Sequence[str],
    use_bedtools: bool,
) -> List[Tuple[str, str]]:
    allowed = set(label_filter)
    rows: List[Tuple[str, str]] = []
    missing = 0
    if use_bedtools:
        filtered = [r for r in records if not allowed or r.labels[0] in allowed]
        for record, seq in bedtools_getfasta(fasta.fasta_path, filtered):
            rows.append((seq, record.labels[0]))
    else:
        for record in records:
            label = record.labels[0]
            if allowed and label not in allowed:
                continue
            try:
                seq = fasta.fetch(record.chrom, record.start, record.end)
            except (KeyError, ValueError):
                missing += 1
                continue
            rows.append((seq, label))
    if missing:
        print(f"Skipped {missing} records with missing/invalid coordinates.")
    return rows


def split_rows(
    rows: Sequence[Tuple[str, str]], train_ratio: float, val_ratio: float
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    total = len(rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return list(rows[:train_end]), list(rows[train_end:val_end]), list(rows[val_end:])


def write_csv(path: Path, rows: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        for text, label in rows:
            writer.writerow([text, label])


def collect_label_set(records: Iterable[PeakRecord]) -> List[str]:
    labels = sorted({label for record in records for label in record.labels})
    return labels


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    human_raw = list(
        iter_peak_records(
            args.human_tsv, coord_col="hg38_coord", label_col=args.label_col, chrom_filter=args.chrom
        )
    )
    mouse_raw = list(
        iter_peak_records(
            args.mouse_tsv, coord_col="coord", label_col=args.label_col, chrom_filter=args.chrom
        )
    )

    human_records = [r for record in human_raw for r in expand_labels(record, args.multi_label)]
    mouse_records = [r for record in mouse_raw for r in expand_labels(record, args.multi_label)]

    human_labels = collect_label_set(human_records)
    mouse_labels = collect_label_set(mouse_records)
    shared_labels = sorted(set(human_labels).intersection(mouse_labels))
    label_filter = shared_labels if args.shared_labels_only else []

    human_fasta = FastaIndex(args.human_fasta)
    if not args.skip_mouse_test and args.mouse_fasta is None:
        raise ValueError("--mouse-fasta is required unless --skip-mouse-test is set.")

    human_rows = load_rows(human_records, human_fasta, label_filter, args.use_bedtools)
    if args.skip_mouse_test:
        mouse_rows: List[Tuple[str, str]] = []
    else:
        mouse_fasta = FastaIndex(args.mouse_fasta)
        mouse_rows = load_rows(mouse_records, mouse_fasta, label_filter, args.use_bedtools)

    print(
        f"Human rows after label handling/filtering: {len(human_rows)} "
        f"(train_ratio={args.train_ratio}, val_ratio={args.val_ratio})"
    )

    rng.shuffle(human_rows)
    train_rows, val_rows, _ = split_rows(human_rows, args.train_ratio, args.val_ratio)

    if args.match_mouse_test_to_dev and mouse_rows:
        target = len(val_rows)
        if target and len(mouse_rows) > target:
            mouse_rows = rng.sample(mouse_rows, target)

    write_csv(args.output_dir / "train.csv", train_rows)
    write_csv(args.output_dir / "dev.csv", val_rows)
    write_csv(args.output_dir / "test.csv", mouse_rows)

    print(
        f"Labels used: {len(label_filter) if label_filter else len(set(human_labels))} "
        f"(shared_only={args.shared_labels_only})"
    )
    print(
        f"Wrote {len(train_rows)} train, {len(val_rows)} dev (human) and "
        f"{len(mouse_rows)} test (mouse) examples to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
