import argparse
import os

import pandas as pd
import pysam
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer


def train_bpe(sequences, vocab_size, min_frequency):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=min_frequency,
        initial_alphabet=list("ACGTN"),
    )
    print(f"Training BPE vocab_size={vocab_size} on {len(sequences):,} sequences")
    tokenizer.train_from_iterator(sequences, trainer)
    return tokenizer


def extract_sequences_by_category(fasta_path, segments_dir):
    fasta = pysam.FastaFile(fasta_path)

    conserved = []
    neutral = []
    accelerated = []

    canonical = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

    for chrm in fasta.references:
        if chrm not in canonical:
            continue
        seg_path = os.path.join(segments_dir, f"{chrm}_phylop_segment.csv")
        if not os.path.exists(seg_path):
            print(f"Warning: missing segments file for {chrm}: {seg_path}")
            continue
        chr_segments = pd.read_csv(seg_path)
        print(chrm)
        if chr_segments.empty:
            continue
        for _, row in chr_segments.iterrows():
            seq = fasta.fetch(
                chrm, int(row["genomic_start"]), int(row["genomic_end"])
            )
            cat = row["category"]
            if cat == "conserved":
                conserved.append(seq)
            elif cat == "neutral":
                neutral.append(seq)
            elif cat == "accelerated":
                accelerated.append(seq)
            else:
                print(f"Warning: unknown category {cat}")

        print(f"conserved region: {len(conserved)}")
        print(f"neutral region: {len(neutral)}")
        print(f"accelerated region: {len(accelerated)}")

    return conserved, neutral, accelerated


def build_merge_unigram(con_path, neu_path, acc_path, vocab_size, output_path):
    conserved_bpe = Tokenizer.from_file(con_path)
    neutral_bpe = Tokenizer.from_file(neu_path)
    accelerated_bpe = Tokenizer.from_file(acc_path)

    v_con = conserved_bpe.get_vocab()
    v_neu = neutral_bpe.get_vocab()
    v_acc = accelerated_bpe.get_vocab()

    vocab_imp = set(v_con.keys())
    vocab_med = set(v_neu.keys())
    vocab_non = set(v_acc.keys())

    all_three = vocab_imp & vocab_med & vocab_non
    print(f"all_three: {len(all_three)}")
    imp_med_only = (vocab_imp & vocab_med) - vocab_non
    print(f"imp_med_only: {len(imp_med_only)}")
    unique_imp = vocab_imp - vocab_med - vocab_non
    print(f"unique_imp: {len(unique_imp)}")
    unique_med = vocab_med - vocab_imp - vocab_non
    print(f"unique_med: {len(unique_med)}")

    final_vocab_list = []
    final_vocab_list.extend(list(all_three))
    final_vocab_list.extend(list(unique_imp))
    final_vocab_list.extend(list(imp_med_only))

    remaining_slots = vocab_size - len(final_vocab_list)
    print(f"remaining_slots: {remaining_slots}")

    if remaining_slots < 0:
        final_vocab_list = final_vocab_list[:vocab_size]
        remaining_slots = 0

    unique_med_tokens = list(unique_med)
    unique_med_tokens.sort(key=lambda token: v_neu[token])
    final_vocab_list.extend(unique_med_tokens[:remaining_slots])

    essentials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "A", "T", "C", "G", "N"]
    for token in essentials:
        if token not in final_vocab_list:
            final_vocab_list.append(token)

    final_vocab_list = final_vocab_list[:vocab_size]
    final_vocab_map = {token: i for i, token in enumerate(final_vocab_list)}

    vocab_entries = []
    for token in final_vocab_map.keys():
        score = len(token) ** 2
        vocab_entries.append((token, score))

    tokenizer = Tokenizer(Unigram(vocab_entries))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train len2 BPE tokenizers and merge into Unigram",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, help="Reference FASTA path")
    parser.add_argument(
        "--segments-dir",
        required=True,
        help="Directory containing per-chromosome *_phylop_segment.csv files",
    )
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        required=True,
        help="One or more vocab sizes",
    )
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory for vocab-specific subfolders",
    )

    args = parser.parse_args()

    conserved, neutral, accelerated = extract_sequences_by_category(
        args.fasta, args.segments_dir
    )

    print(f"conserved region: {len(conserved)}")
    print(f"neutral region: {len(neutral)}")
    print(f"accelerated region: {len(accelerated)}")


    for vocab_size in args.vocab_sizes:
        out_dir = os.path.join(args.output_dir, f"vocab_{vocab_size}")
        os.makedirs(out_dir, exist_ok=True)


        con_tok = train_bpe(conserved, vocab_size, args.min_frequency)
        neu_tok = train_bpe(neutral, vocab_size, args.min_frequency)
        acc_tok = train_bpe(accelerated, vocab_size, args.min_frequency)

        con_path = os.path.join(out_dir, "con_tokenizer.json")
        neu_path = os.path.join(out_dir, "neu_tokenizer.json")
        acc_path = os.path.join(out_dir, "acc_tokenizer.json")
        con_tok.save(con_path)
        neu_tok.save(neu_path)
        acc_tok.save(acc_path)

        merge_path = os.path.join(out_dir, "merge_tokenizer_unigram_len2.json")
        build_merge_unigram(con_path, neu_path, acc_path, vocab_size, merge_path)


if __name__ == "__main__":