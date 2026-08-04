#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-region token importance across promoter, enhancer, exon and tokenizer variants."
        )
    )
    parser.add_argument("--fasta", required=True, help="Genome FASTA path")
    parser.add_argument("--promoter-bed", required=True, help="Promoter BED path")
    parser.add_argument("--enhancer-bed", required=True, help="Enhancer BED path")
    parser.add_argument("--exon-bed", required=True, help="Exon BED path")
    parser.add_argument(
        "--out-dir",
        default="single_region_importance_all4",
        help="Output directory",
    )
    parser.add_argument(
        "--vocabs",
        default="2048,3072,4096,5120",
        help="Comma-separated vocab sizes",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=50000,
        help="Reservoir-sample up to this many regions per region set",
    )
    parser.add_argument(
        "--num-shuffles",
        type=int,
        default=10,
        help="Number of shuffled-background replicates",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--keep-n", action="store_true", help="Keep sequences with non-ACGT")
    return parser.parse_args()


def tokenizer_configs(root: Path, vocabs: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for vocab in vocabs:
        out.append(
            {
                "model": "baseline",
                "vocab": vocab,
                "tokenizer": str(
                    root
                    / "tokenizer_evaluation"
                    / "baseline_bpe"
                    / f"vocab_{vocab}"
                    / f"{vocab}_tokenizer.json"
                ),
            }
        )
        out.append(
            {
                "model": "merge_len2",
                "vocab": vocab,
                "tokenizer": str(
                    root
                    / "tokenizer_evaluation"
                    / "merge_bpe"
                    / f"vocab_{vocab}"
                    / "merge_tokenizer_unigram_len2.json"
                ),
            }
        )
    return out


def read_summary(sig_path: Path, all_path: Path) -> Dict[str, str]:
    important = 0
    top_token = ""
    top_log2 = None

    if sig_path.exists():
        with sig_path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                important += 1

    if all_path.exists():
        with all_path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    q = float(row.get("fdr_bh", "1"))
                    l2 = float(row.get("log2_enrichment_observed_over_bg", "0"))
                except ValueError:
                    continue
                if l2 <= 0:
                    continue
                if top_log2 is None or (q < top_log2[0]) or (q == top_log2[0] and l2 > top_log2[1]):
                    top_log2 = (q, l2)
                    top_token = row.get("token", "")

    return {
        "important_tokens": important,
        "top_important_token": top_token,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = {
        "promoter": args.promoter_bed,
        "enhancer": args.enhancer_bed,
        "exon": args.exon_bed,
    }

    allowed = {"2048", "3072", "4096", "5120"}
    vocabs = [v.strip() for v in args.vocabs.split(",") if v.strip()]
    bad = [v for v in vocabs if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported vocab(s): {bad}. Allowed: {sorted(allowed)}")

    script = root / "single_region_token_importance.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    summary_rows = []
    for cfg in tokenizer_configs(root, vocabs):
        for region_name, bed_path in regions.items():
            prefix = out_dir / f"importance_{region_name}_{cfg['model']}_{cfg['vocab']}"
            cmd = [
                sys.executable,
                str(script),
                "--fasta",
                args.fasta,
                "--region-bed",
                bed_path,
                "--tokenizer",
                cfg["tokenizer"],
                "--region-name",
                region_name,
                "--out-prefix",
                str(prefix),
                "--max-regions",
                str(args.max_regions),
                "--num-shuffles",
                str(args.num_shuffles),
                "--seed",
                str(args.seed),
            ]
            if args.keep_n:
                cmd.append("--keep-n")

            print(f"\nRunning: {region_name} | {cfg['model']} | vocab={cfg['vocab']}")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)

            all_path = Path(f"{prefix}.all_tokens.tsv")
            sig_path = Path(f"{prefix}.important_fdr_0.05.tsv")
            summary = read_summary(sig_path, all_path)
            summary_rows.append(
                {
                    "region": region_name,
                    "model": cfg["model"],
                    "vocab": cfg["vocab"],
                    "tokenizer_path": cfg["tokenizer"],
                    "all_tokens_tsv": str(all_path),
                    "important_tokens_tsv": str(sig_path),
                    **summary,
                }
            )

    out_summary = out_dir / "summary_single_region_importance.tsv"
    with out_summary.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "region",
                "model",
                "vocab",
                "tokenizer_path",
                "all_tokens_tsv",
                "important_tokens_tsv",
                "important_tokens",
                "top_important_token",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {out_summary}")


if __name__ == "__main__":
    main()
