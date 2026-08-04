import os
import random
from os.path import join
from collections import Counter
import numpy as np
import pysam
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers import models, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import Split


def writetsv(data, label, savefile):
    with open(savefile, 'w') as f:
        f.write('sequence\tlabels\n')
        for seq, lab in zip(data, label):
            f.write(f'{seq}\t{lab}\n')


def nonoverlap_split(tokens, maxlen, tolerance=0.5):
    seqs = []
    skipped = 0

    num_windows = len(tokens) // maxlen

    for i in range(num_windows):
        window = tokens[i*maxlen:(i+1)*maxlen]

        # NEW: token-aware N detection
        num_N = sum('N' in tok for tok in window)

        if num_N / maxlen < tolerance:
            seqs.append(" ".join(window))
        else:
            skipped += 1

    print(f"In this chromosome, skipped sequences: {skipped}")
    return seqs


def tokenize_full_sequence_collect(tokenizer, sequence, chunk_size=1_000_000):
    raw_tokens = []

    for i in range(0, len(sequence), chunk_size):
        chunk = sequence[i:i + chunk_size]
        encoded = tokenizer.encode(chunk)
        raw_tokens.extend(encoded.tokens)

        if i % (10 * chunk_size) == 0:
            print(f"Processed {i:,} bp")

    return raw_tokens

maxlen = 512
tolerance = 0.5
CHUNK_SIZE = 1_000_000
fasta_path = os.environ.get('EVOLEN_FASTA', os.path.expanduser('~/evolen_data/hg38.fa'))
args_token_path = os.environ.get('EVOLEN_TOKENS_OUT', os.path.expanduser('~/evolen_data/output_tokens'))
os.makedirs(args_token_path, exist_ok=True)

# Set to a list like ["chr1", "chr2"] to limit processing.
CHROMOSOMES = None
EXCLUDE_CHROMS = set()



# --- 2. LOAD YOUR TOKENIZERS ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_PATHS = {
    "merge_bpe_5120":   os.path.join(_REPO, "assets/tokenizers/merge_tokenizer_unigram_len2.json"),
    "baseline_bpe_5120": os.path.join(_REPO, "assets/tokenizers/5120_tokenizer.json"),
}

tokenizers = {}
for name, path in VOCAB_PATHS.items():
    tokenizers[name] = Tokenizer.from_file(path)


with pysam.FastaFile(fasta_path) as genome:
    chroms = genome.references if CHROMOSOMES is None else CHROMOSOMES
    chroms = [c for c in chroms if c not in EXCLUDE_CHROMS]

    for tok_name, tok in tokenizers.items():
        print(tok.pre_tokenizer)
        print(tok.model)

        print(f"\n=== Processing tokenizer: {tok_name} ===")

        all_seqs = []
        all_labels = []

        for chrm in chroms:
            full_sequence = genome.fetch(reference=chrm)

            print(f"\nChromosome: {chrm}")
            print(f"Total length: {len(full_sequence):,} bases")
            print(f"First 100 bases:\n{full_sequence[:100]}")

            # 1. Tokenize full chromosome
            raw_tokens = tokenize_full_sequence_collect(
                tok,
                full_sequence,
                chunk_size=CHUNK_SIZE
            )
            print(f"Total raw tokens: {len(raw_tokens):,}")

            # 2. Build sequences
            final_seqs = nonoverlap_split(
                tokens=raw_tokens,
                maxlen=maxlen,
                tolerance=tolerance
            )

            print(f"Total sequences for pretrain: {len(final_seqs):,}")

            all_seqs.extend(final_seqs)
            all_labels.extend([chrm] * len(final_seqs))

        if not all_seqs:
            print(f"No sequences generated for tokenizer: {tok_name}")
            continue

        # 3. Shuffle
        combined = list(zip(all_seqs, all_labels))
        random.seed(42)
        random.shuffle(combined)
        shuffle_data, shuffle_labels = zip(*combined)

        # 4. Train / Val split
        train_num = int(0.9 * len(shuffle_data))

        train_data = shuffle_data[:train_num]
        train_labels = shuffle_labels[:train_num]
        val_data = shuffle_data[train_num:]
        val_labels = shuffle_labels[train_num:]

        # 5. Save TSVs
        train_path = join(
            args_token_path,
            f"{tok_name}_allchr_all_tokenized_train.tsv"
        )
        val_path = join(
            args_token_path,
            f"{tok_name}_allchr_all_tokenized_val.tsv"
        )

        writetsv(train_data, train_labels, train_path)
        writetsv(val_data, val_labels, val_path)

        print(f"Saved:\n  {train_path}\n  {val_path}")
