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
            "Run token enrichment analysis across tokenizer variants "
            "(baseline + merge_len2; vocab 3072 + 5120)."
        )
    )
    parser.add_argument("--fasta", required=True, help="Genome FASTA path")
    parser.add_argument("--promoter-bed", required=True, help="Promoter BED path")
    parser.add_argument("--enhancer-bed", required=True, help="Enhancer BED path")
    parser.add_argument(
        "--out-dir",
        default="token_enrichment_variants",
        help="Directory to store all outputs",
    )
    parser.add_argument(
        "--max-regions-per-set",
        type=int,
        default=100000,
        help="Reservoir-sample up to this many intervals per set",
    )
    parser.add_argument(
        "--vocabs",
        default="2048,3072,4096,5120",
        help="Comma-separated vocab sizes to run (subset of: 2048,3072,4096,5120)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--keep-n",
        action="store_true",
        help="Keep sequences with N characters",
    )
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


def read_sig_summary(sig_path: Path) -> Dict[str, int]:
    total = 0
    prom = 0
    enh = 0
    if not sig_path.exists():
        return {"sig_tokens": 0, "sig_promoter": 0, "sig_enhancer": 0}

    with sig_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            if row.get("enriched_in") == "promoter":
                prom += 1
            elif row.get("enriched_in") == "enhancer":
                enh += 1
    return {"sig_tokens": total, "sig_promoter": prom, "sig_enhancer": enh}


def read_best_tokens(all_path: Path) -> Dict[str, str]:
    best_prom = ""
    best_enh = ""
    best_prom_fdr = None
    best_enh_fdr = None

    if not all_path.exists():
        return {"best_promoter_token": "", "best_enhancer_token": ""}

    with all_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            side = row.get("enriched_in", "")
            try:
                fdr = float(row.get("fdr_bh", "1"))
            except ValueError:
                fdr = 1.0
            tok = row.get("token", "")
            if side == "promoter":
                if best_prom_fdr is None or fdr < best_prom_fdr:
                    best_prom_fdr = fdr
                    best_prom = tok
            elif side == "enhancer":
                if best_enh_fdr is None or fdr < best_enh_fdr:
                    best_enh_fdr = fdr
                    best_enh = tok
    return {"best_promoter_token": best_prom, "best_enhancer_token": best_enh}


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"2048", "3072", "4096", "5120"}
    vocabs = [v.strip() for v in args.vocabs.split(",") if v.strip()]
    if not vocabs:
        raise ValueError("No vocab provided in --vocabs")
    bad = [v for v in vocabs if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported vocab(s): {bad}. Allowed: {sorted(allowed)}")

    script = root / "token_enrichment_promoter_enhancer.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    summary_rows = []
    configs = tokenizer_configs(root, vocabs)

    for cfg in configs:
        tok_path = Path(cfg["tokenizer"])
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

        prefix = out_dir / f"token_enrichment_{cfg['model']}_{cfg['vocab']}"
        cmd = [
            sys.executable,
            str(script),
            "--fasta",
            args.fasta,
            "--promoter-bed",
            args.promoter_bed,
            "--enhancer-bed",
            args.enhancer_bed,
            "--tokenizer",
            str(tok_path),
            "--out-prefix",
            str(prefix),
            "--max-regions-per-set",
            str(args.max_regions_per_set),
            "--seed",
            str(args.seed),
        ]
        if args.keep_n:
            cmd.append("--keep-n")

        print(f"\nRunning: {cfg['model']} vocab={cfg['vocab']}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        all_path = Path(f"{prefix}.all_tokens.tsv")
        sig_path = Path(f"{prefix}.sig_fdr_0.05.tsv")
        sig_summary = read_sig_summary(sig_path)
        best_tokens = read_best_tokens(all_path)

        summary_rows.append(
            {
                "model": cfg["model"],
                "vocab": cfg["vocab"],
                "tokenizer_path": str(tok_path),
                "all_tokens_tsv": str(all_path),
                "sig_tokens_tsv": str(sig_path),
                **sig_summary,
                **best_tokens,
            }
        )

    summary_path = out_dir / "summary_variants.tsv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "vocab",
                "tokenizer_path",
                "all_tokens_tsv",
                "sig_tokens_tsv",
                "sig_tokens",
                "sig_promoter",
                "sig_enhancer",
                "best_promoter_token",
                "best_enhancer_token",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
