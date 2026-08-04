import argparse
import os

import pysam
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def load_bounds_from_tsv(path):
    bounds = {}
    with open(path) as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            cols = line.split()
            if len(cols) < 3:
                continue
            chrm = cols[0]
            try:
                min_start = int(cols[1])
                max_end = int(cols[2])
            except ValueError:
                continue
            bounds[chrm] = (min_start, max_end)
    return bounds


def build_seq_list(fasta_path, bedgraph_dir, chunk_size):
    bounds_path = os.path.join(bedgraph_dir, "bedgraph_bounds.tsv")
    chr_bounds = load_bounds_from_tsv(bounds_path)
    print(chr_bounds)

    all_seq_list = []
    with pysam.FastaFile(fasta_path) as genome:
        for chrm, (min_start, max_end) in chr_bounds.items():
            print(chrm)
            if chrm not in genome.references:
                continue
            ref_len = genome.get_reference_length(chrm)
            min_start = min(min_start, ref_len)
            print(min_start)
            max_end = min(max_end, ref_len)
            print(max_end)
            if max_end <= min_start:
                continue
            full_sequence = genome.fetch(
                reference=chrm, start=min_start, end=max_end
            )

            print(f"Total length: {len(full_sequence):,} bases")
            print(f"First 100 bases:\n{full_sequence[:100]}")

            seq_list = [
                full_sequence[i : i + chunk_size]
                for i in range(0, len(full_sequence), chunk_size)
                if "N" not in full_sequence[i : i + chunk_size]
            ]
            all_seq_list.extend(seq_list)

    return all_seq_list


def train_bpe_for_vocab(sequences, vocab_size, output_dir, min_frequency):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=min_frequency,
        initial_alphabet=["A", "C", "G", "T", "N"],
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Training BPE vocab_size={vocab_size} on {len(sequences):,} chunks")
    tokenizer.train_from_iterator(sequences, trainer)
    tokenizer.save(os.path.join(output_dir, f"{vocab_size}_tokenizer.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline BPE tokenizers for multiple vocab sizes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, help="Reference FASTA path")
    parser.add_argument("--bedgraph-dir", required=True, help="bedGraph directory")
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        required=True,
        help="One or more vocab sizes",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory for vocab-specific subfolders",
    )

    args = parser.parse_args()

    sequences = build_seq_list(args.fasta, args.bedgraph_dir, args.chunk_size)

    for vocab_size in args.vocab_sizes:
        
        out_dir = os.path.join(args.output_dir, f"vocab_{vocab_size}")
        print(f"save baseline bpe with vocab_{vocab_size} to {out_dir}")
        train_bpe_for_vocab(
            sequences, vocab_size, out_dir, min_frequency=args.min_frequency
        )


if __name__ == "__main__":
    main()
