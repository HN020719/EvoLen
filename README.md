# EvoLen Tokenizer

Code for **EvoLen: Evolution-Guided Tokenization for DNA Language Model**,
accepted at **COLM 2026**. A preprint is available at
[arXiv:2604.08698](https://arxiv.org/abs/2604.08698).

EvoLen is a DNA tokenizer that combines evolutionary stratification with length-aware
decoding: it partitions the genome by phyloP conservation, trains a BPE vocabulary on each
category, merges them under a conservation-prioritised rule, and decodes with a
length-squared scoring function so that motif-scale units survive as single tokens.

This repository contains the tokenizer construction code, pretraining and fine-tuning
pipelines, token analysis, and experiment configuration. Pretrained and fine-tuned
checkpoints are hosted on Hugging Face.

## Models and Datasets

The EvoLen and baseline models and datasets are hosted on [Hugging Face](https://huggingface.co/EvoLenTokenizer).

## Repository Structure

```text
evolen/
├── assets/                    # Tokenizers and unified experiment manifest
├── tokenizer/                 # EvoLen tokenizer construction and ablation variants
├── pretrain/                  # Corpus preparation and MLM pretraining (100k / 200k steps)
├── data_prep/                 # Builders for the derived cCRE and ATAC benchmarks
├── analysis/                  # Token analysis: motif, phyloP and enrichment (paper Section 4)
├── train_scripts/             # Benchmark-specific Python scripts for model training
├── sft_evolen.py              # Unified CLI for selecting and running EvoLen SFT benchmark tasks
├── run_benchmark.sh           # Runs the selected fine-tuning task
├── environment.yml            # Conda environment and dependencies required to run EvoLen SFT
└── README.md
```

## Tokenizer Construction

The tokenizers in `assets/tokenizers/` are released ready to use, so this section is only
needed to rebuild them from scratch or to reproduce the ablations.

```text
phyloP bedGraph
  └─ process_bedgraph_all.py         → {chrom}_phylop_segment.csv          (§3.1 stratification)
       ├─ train_len2_bpe_multi.py    → merge_tokenizer_unigram_len2.json   (§3.1–3.3, EvoLen)
       ├─ train_baseline_bpe_multi.py → 5120_tokenizer.json                (baseline control)
       └─ train_ablation_tokenizers.py → 3 ablation tokenizers             (§5.3)
```

**1. Evolutionary stratification.** Bins the genome into non-overlapping 100 bp windows,
averages phyloP per bin, and assigns each bin to `conserved` / `neutral` / `accelerated`
with a two-tailed z-score rule at p < 0.1 (z = 1.645). Adjacent same-category bins are
merged into regions.

```bash
python tokenizer/process_bedgraph_all.py \
  --bedgraph_dir /path/to/phylop_bedgraphs \
  --out_dir /path/to/phylop_segments \
  --bin_size 100 --p_value 0.1
```

**2. EvoLen tokenizer.** Trains a separate BPE on each of the three sequence pools, merges
the vocabularies under the conservation-prioritised rule (tokens shared by all three →
conserved-specific → conserved∩neutral → neutral-specific to fill capacity), then
serialises the result as a Unigram model scored by `score(t) = |t|²` so that decoding
prefers longer, motif-preserving segmentations.

```bash
python tokenizer/train_len2_bpe_multi.py \
  --fasta /path/to/hg38.fa \
  --segments-dir /path/to/phylop_segments \
  --vocab-sizes 2048 3072 4096 5120 \
  --output-dir /path/to/merge_bpe
```

Writes `vocab_<size>/merge_tokenizer_unigram_len2.json` plus the three intermediate
category tokenizers (`con_`, `neu_`, `acc_tokenizer.json`), which step 4 reuses.

**3. Baseline tokenizer.** Standard BPE over the whole genome with no stratification —
the control used for every baseline model.

```bash
python tokenizer/train_baseline_bpe_multi.py \
  --fasta /path/to/hg38.fa \
  --bedgraph-dir /path/to/phylop_bedgraphs \
  --vocab-sizes 2048 3072 4096 5120 \
  --output-dir /path/to/baseline_bpe
```

**4. Ablations.** Each variant removes exactly one EvoLen component:

| variant | removes | keeps |
|---|---|---|
| `no_partition` | phyloP stratification (uses baseline BPE vocab) | length-squared scoring |
| `no_priority`  | conserved-first merge order (random merge) | stratification, length-squared scoring |
| `no_length`    | length-squared scoring (uniform scores) | stratification, priority merge |

```bash
python tokenizer/train_ablation_tokenizers.py \
  --baseline-bpe /path/to/baseline_bpe/vocab_5120/5120_tokenizer.json \
  --category-bpe-dir /path/to/merge_bpe/vocab_5120 \
  --vocab-size 5120 \
  --output-dir /path/to/ablation_tokenizers
```

Pretrained models for the three ablations are on Hugging Face
([no_priority](https://huggingface.co/nancyH/model_ablation_no_priority),
[no_partition](https://huggingface.co/nancyH/model_ablation_no_partition),
[no_length](https://huggingface.co/nancyH/model_ablation_no_length)), and the three
tokenizer files together at
[ablation_tokenizer_5120](https://huggingface.co/nancyH/ablation_tokenizer_5120).

## Pretraining

The released checkpoints can be used directly, so this section is only needed to
pretrain from scratch.

```text
tokenizer JSON
  └─ tok_split_full.py       → *_allchr_all_tokenized_{train,val}_chrOnly.tsv
       └─ 04.pretrain.sh     → 100k-step checkpoint   (run_mlm.py)
            └─ 04.pretrain_200k.sh → 200k-step checkpoint  (run_mlm_200k.py, resumes from 100k)
```

**1. Tokenized corpus.** Already published — download it rather than rebuilding:

```bash
hf download EvoLenTokenizer/pretrain --repo-type dataset --local-dir ~/evolen_data
unzip ~/evolen_data/merge_bpe_5120_allchr_all_tokenized_chrOnly.zip -d ~/evolen_data/output_tokens
```

To rebuild instead, `tok_split_full.py` tokenizes each chromosome in 1 Mbp chunks, splits
into non-overlapping 512-token windows and drops windows where more than half the tokens
contain `N`:

```bash
EVOLEN_FASTA=~/evolen_data/hg38.fa \
EVOLEN_TOKENS_OUT=~/evolen_data/output_tokens \
python pretrain/tok_split_full.py
```

**2. Pretrain to 100k steps.** BERT-base (12 layers, 768 hidden), 512-token sequences,
batch 96 per device, lr 4e-5 with 10k warmup steps, MLM probability 0.15, seed 42:

```bash
EVOLEN_DATA_DIR=~/evolen_data/output_tokens \
EVOLEN_OUT_DIR=~/evolen_models/merge_bpe_5120 \
./pretrain/04.pretrain.sh merge_bpe_5120        # or baseline_bpe_5120
```

**3. Continue to 200k steps.** Resumes from `checkpoint-100000` with identical
hyperparameters. It calls `run_mlm_200k.py`, which is `run_mlm.py` plus a
`torch.load(weights_only=False)` shim required to resume a checkpoint under torch >= 2.6:

```bash
EVOLEN_DATA_DIR=~/evolen_data/output_tokens \
EVOLEN_OUT_DIR=~/evolen_models/merge_bpe_5120 \
./pretrain/04.pretrain_200k.sh merge_bpe_5120
```

The resulting checkpoints are published as
[`evolen-100k`](https://huggingface.co/EvoLenTokenizer/evolen-100k),
[`base-100k`](https://huggingface.co/EvoLenTokenizer/base-100k),
[`evolen-200k`](https://huggingface.co/EvoLenTokenizer/evolen-200k) and
[`base-200k`](https://huggingface.co/EvoLenTokenizer/base-200k).

## Token Analysis

`analysis/` reproduces the token-level evaluations in Section 4 of the paper. These
scripts compute the reported statistics; figure rendering is not included.

| directory | paper section | reported statistic |
|---|---|---|
| `analysis/motif/` | §4.1 motif preservation (P1) | perfect match rate per tokenizer |
| `analysis/length_dist/` | §4.2 regulatory specificity (P2) | token-length bin distributions and pairwise JS |
| `analysis/phylop/` | §4.3 evolutionary consistency (P3) | mean phyloP per token, by conservation category |
| `analysis/enrichment/` | §4.4 pattern recurrence (P4) | mean log2 fold-change vs neutral-intronic background |
| `analysis/diagnostics/` | design-choice diagnostics | stratification window size; pool-BPE distinctness |

`analysis/diagnostics/run_window_diagnostic.py` sweeps the stratification window from
5 bp to 500 bp and reports per-window phyloP dispersion, lag-1 autocorrelation,
label stability against the 100 bp reference (base-level agreement and Cohen's kappa),
and the probability that a 6-12 bp motif crosses a bin boundary — the evidence behind
the choice of 100 bp. `run_pool_bpe_diagnostic.py` reports vocabulary Jaccard overlap
and compression/entropy statistics for the three category-specific BPEs, showing they
are not redundant.

`analysis/length_dist/js_distance.py` reports the pairwise Jensen–Shannon distance
between per-region token-length distributions, and reproduces the values in §4.2 by
default.

### Input data

Download and extract the derived interval files:

```bash
hf download EvoLenTokenizer/analysis-data --repo-type dataset --local-dir ~/evolen_analysis
unzip ~/evolen_analysis/evolen_analysis_data.zip -d ~/evolen_analysis
```

This provides the region BEDs, the ENCODE SCREEN cCRE class BEDs, and the thresholded
JASPAR 2024 motif list. The hg38 reference and the phyloP scores are public reference
data and are **not** bundled — the
[dataset card](https://huggingface.co/datasets/EvoLenTokenizer/analysis-data) gives the
UCSC download commands for both.

Analysis scripts resolve their paths from a single environment variable:

```bash
export EVOLEN_ROOT=/path/to/your/data_root
```

### Tokenizer coverage

`assets/tokenizers/` ships the **vocab 5,120** tokenizers only — the size used for every
downstream experiment. The P1 and P3 analyses additionally sweep vocab 2,048, 3,072 and
4,096, so those scripts reference `vocab_2048/`, `vocab_3072/` and `vocab_4096/`
directories that are not included. Rebuild them with the construction scripts:

```bash
python tokenizer/train_len2_bpe_multi.py     --vocab-sizes 2048 3072 4096 5120 ...
python tokenizer/train_baseline_bpe_multi.py --vocab-sizes 2048 3072 4096 5120 ...
```

Point the analysis scripts at the resulting `--output-dir`. Running them against only
vocab 5,120 reproduces the 5,120 column of Figure 2A and 2C; the other three vocab sizes
require the rebuild.

These scripts cover the EvoLen tokenizer, the baseline BPE control and the §5.3 ablations.
The §5.2 comparison against other tokenization strategies (DNABERT-2, GROVER) is not
included.

Per-token phyloP aggregates used for Figure 2C are also published directly at
[nancyH/token_evaluation](https://huggingface.co/datasets/nancyH/token_evaluation), so
Figure 2C can be re-plotted without recomputing the genome-wide scan.

## Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate evolen
```

## Training Scripts

Training scripts are located in `train_scripts/`:

* ATAC_train.py
* GBM_train.py
* GUE_train.py
* NT_train.py
* screen_train.py

These scripts correspond to different downstream tasks and are launched automatically by `sft_evolen.py`.

## Downloading Benchmark Data

Benchmark datasets are hosted under the
[EvoLenTokenizer Hugging Face organization](https://huggingface.co/EvoLenTokenizer).

Download the benchmark archives:

```bash
mkdir -p ~/evolen_data/downloads

hf download EvoLenTokenizer/NT-benchmarks --repo-type dataset --local-dir ~/evolen_data/downloads/NT
hf download EvoLenTokenizer/GUE-benchmarks --repo-type dataset --local-dir ~/evolen_data/downloads/GUE
hf download EvoLenTokenizer/GBM-benchmarks --repo-type dataset --local-dir ~/evolen_data/downloads/GBM
hf download EvoLenTokenizer/ATAC-benchmarks --repo-type dataset --local-dir ~/evolen_data/downloads/ATAC
hf download EvoLenTokenizer/Screen-benchmarks --repo-type dataset --local-dir ~/evolen_data/downloads/SCREEN
```

Extract the downloaded archives:

```bash
mkdir -p ~/evolen_data/{NT,GUE,GBM,ATAC,SCREEN}

unzip ~/evolen_data/downloads/NT/*.zip -d ~/evolen_data/NT
unzip ~/evolen_data/downloads/GUE/*.zip -d ~/evolen_data/GUE
unzip ~/evolen_data/downloads/GBM/*.zip -d ~/evolen_data/GBM
unzip ~/evolen_data/downloads/ATAC/*.zip -d ~/evolen_data/ATAC
unzip ~/evolen_data/downloads/SCREEN/*.zip -d ~/evolen_data/SCREEN
```

When running `sft_evolen.py`, set `--data-path` to the extracted benchmark root containing the task directories. The CLI automatically selects the requested task and appends `split/` for NT and GBM.

### How the benchmark splits were produced

The archives above are ready to use; this section records how they were derived.

**GUE** (27 tasks) — used exactly as distributed by DNABERT-2. Each task already ships
`train.csv` / `dev.csv` / `test.csv`, and no reprocessing was applied.

**GenomicBenchmarks** (9 tasks) — downloaded through the `datasets` library, then
re-split, because the original release provides only train and test. All available splits
are concatenated, shuffled with seed 42, and divided 80/10/10 into train/dev/test, written
as tab-separated `sequence` / `labels` columns under `<task>/split/`.

**Nucleotide Transformer** (18 tasks) — the distributed train/test pair was likewise
divided to carve out a development set, giving `<task>/split/{train,dev,test}.csv`.

**Multi-SCREEN** (`allchr_csv`, 8-way cCRE classification) — built by
[`data_prep/prepare_ccre_finetuning.py`](data_prep/prepare_ccre_finetuning.py) from the
per-class ENCODE SCREEN cCRE FASTAs, downsampled so every class matches the smallest
(26,102 elements), then split 80/10/10 at seed 42.

**Multi-ATAC** (`human_mouse_superclass_allchr`, 4-way EXC/INH/GLIA/VASC) — built by
[`data_prep/prepare_human_mouse_finetune.py`](data_prep/prepare_human_mouse_finetune.py)
from the human and mouse snATAC-seq peak tables, extracting sequence against hg38 and
mm10:

```bash
python data_prep/prepare_human_mouse_finetune.py \
  --human-tsv Supplementary_Table_4_Human_ATAC_peaks.tsv \
  --mouse-tsv Supplementary_Table_7_Mouse_ATAC_peaks.tsv \
  --label-col superclass \
  --human-fasta hg38.fa --mouse-fasta mm10.fa \
  --output-dir <out>/human_mouse_superclass_allchr
```

Note the `--label-col superclass` override: the script defaults to `celltype`, which
produces the finer-grained task rather than the four superclasses reported in the paper.

## Example Usage

```bash
cd ~/evolen

python sft_evolen.py \
  --model evolen \
  --scale 100k \
  --benchmark ATAC \
  --task human_mouse_superclass_allchr \
  --data-path ~/evolen_data/ATAC \
  --output-path ~/sft_outputs \
  --project-name EvoLen-ATAC \
  --gpu-id 0 \
  --wandb
```

`--data-path` should point to the root directory of the extracted benchmark dataset. The CLI automatically resolves the selected task directory and appends split/ for NT and GBM tasks.

View all available options with:

```bash
python sft_evolen.py --help
```

## Optional Weights & Biases Logging

Weights & Biases logging is optional and disabled by default. To enable it, first log in with your own W&B account:

```bash
wandb login
```

or set your API key as an environment variable:

```bash
export WANDB_API_KEY=your_key_here
```

Then include `--wandb` when running `sft_evolen.py`.

## Running Long Jobs with `tmux`

Start a session:

```bash
tmux new -s evolen_run
```

Run your training command, then detach from tmux session with Ctrl+B then press D.

Reconnect later with:

```bash
tmux attach -t evolen_run
```

## Additional Methods Mentioned in the Paper

The primary EvoLen and baseline checkpoints are hosted on Hugging Face. Additional models associated with EvoLen are available on [Zenodo](https://zenodo.org/records/21365247).

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use EvoLen, please cite the paper:

```bibtex
@inproceedings{huang2026evolen,
  title     = {EvoLen: Evolution-Guided Tokenization for DNA Language Model},
  author    = {Huang, Nan and Zhou, Xiaoxiao and Cui, Junxia and
               Tapia-Pacheco, Mario and Amariuta, Tiffany and Li, Yang and
               Shang, Jingbo},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

The preprint is also on arXiv:

```bibtex
@article{huang2026evolen_arxiv,
  title   = {EvoLen: Evolution-Guided Tokenization for DNA Language Model},
  author  = {Huang, Nan and Zhou, Xiaoxiao and Cui, Junxia and
             Tapia-Pacheco, Mario and Amariuta, Tiffany and Li, Yang and
             Shang, Jingbo},
  journal = {arXiv preprint arXiv:2604.08698},
  year    = {2026}
}
```
