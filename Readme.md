# Rep vs. Arch in Neural Differential Distinguisher

Codebase for experiments on **Representation vs. Architecture in Neural Differential Distinguishers**.

This repository generates binary classification datasets from reduced-round block ciphers, trains several neural distinguishers on those datasets, and includes a row-ablation script for studying how much of the signal comes from the chosen representation rather than the model architecture.

## What the project contains

- Dataset generators for four 64-bit block ciphers:
  - `GIFT-64` with a 128-bit key
  - `PRESENT-64` with an 80-bit key
  - `RECTANGLE-64` with an 80-bit key
  - `SPECK-64/128`
- Shared neural models:
  - Logistic regression
  - MLP
  - 1D CNN
  - 1D ResNet
- A shared training script for `.npz` datasets
- A row-ablation analysis script for multi-difference (`R2`) datasets
- Helper files for fixed plaintext generation and reference differential sets

## Repository layout

```text
.
├── Ciphers/
│   ├── Gift-64/
│   │   ├── dataset_generator.py
│   │   ├── gift64_cipher.py
│   │   └── plaintexts64.txt
│   ├── Present-64/
│   │   ├── dataset_generator.py
│   │   ├── present64_cipher.py
│   │   └── plaintexts64.txt
│   ├── Rectangle-64/
│   │   ├── dataset_generator.py
│   │   ├── rectangle64_cipher.py
│   │   └── plaintexts64.txt
│   └── Speck-64/
│       ├── dataset_generator.py
│       ├── speck64_cipher.py
│       └── plaintexts64.txt
├── Helpers/
│   └── plaintext_generator.py
├── Info/
│   └── differences.txt
├── Models/
│   ├── models.py
│   └── train.py
├── Other/
│   └── rep_analysis_row_ablation.py
└── requirements.txt
```

## Environment

Recommended:

- Python 3.10+
- `numpy`
- `torch`
- `matplotlib`
- `scikit-learn`

Install with:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you prefer not to activate the environment, use `venv/bin/python` directly in the commands below.

## Core idea

The project builds binary datasets with:

- `y = 1`: ciphertext differences produced from structured plaintext pairs or tuples
- `y = 0`: ciphertext differences produced from unrelated random plaintexts

Each sample is stored as a bit matrix:

- `R1` datasets: shape `(N, 1, 64)`
- `R2` datasets: shape `(N, num_rows, 64)`

For multi-difference datasets, each row is a 64-bit ciphertext-difference bit vector. In the current generators, `num_rows = len(deltas) - 1` because the first delta is used as the reference row.

Datasets are saved as compressed `.npz` files containing at least:

- `X`: bit tensor
- `y`: binary labels

They also include metadata such as `rounds`, `seed`, key material, and either `delta` or `deltas`.

## Plaintexts and differential sets

Each cipher folder already contains a `plaintexts64.txt` file. The dataset generators will use it when present and fall back to random plaintexts if the file is missing.

To generate a fresh plaintext file:

```bash
python3 Helpers/plaintext_generator.py
```

Reference differential choices used in the project are listed in `Info/differences.txt`, including:

- single-difference setting `0x1`
- a main multi-difference set beginning with `0x0,0x1,0x2,...`
- an additional single- and multi-difference configuration for subsection-style experiments

## Generating datasets

All cipher generators share the same CLI:

```bash
python3 Ciphers/<Cipher>/dataset_generator.py \
  --mode {single|multi|both} \
  --rounds <r> \
  --n_real <count> \
  --n_random <count> \
  [--delta <hex>] \
  [--deltas <hex1,hex2,...>] \
  [--plaintext_file plaintexts64.txt] \
  [--seed <int>] \
  [--key <hex>]
```

Notes:

- `--mode single` creates an `R1` dataset.
- `--mode multi` creates an `R2` dataset.
- `--mode both` writes both files in one run.
- If `--seed` or `--key` is omitted, the generator creates fresh random values for that execution and prints them.

### Output filenames

The current naming convention is:

- `gift64_R1_r<rounds>_s<seed>.npz`
- `gift64_R2_r<rounds>_s<seed>.npz`
- `present64_R1_r<rounds>_s<seed>.npz`
- `present64_R2_r<rounds>_s<seed>.npz`
- `rect64_R1_r<rounds>_s<seed>.npz`
- `rect64_R2_r<rounds>_s<seed>.npz`
- `speck64_R1_r<rounds>_s<seed>.npz`
- `speck64_R2_r<rounds>_s<seed>.npz`

### Example: RECTANGLE-64

Single-difference (`R1`) dataset:

```bash
python3 Ciphers/Rectangle-64/dataset_generator.py \
  --mode single \
  --rounds 6 \
  --n_real 50000 \
  --n_random 50000 \
  --delta 0x1
```

Multi-difference (`R2`) dataset:

```bash
python3 Ciphers/Rectangle-64/dataset_generator.py \
  --mode multi \
  --rounds 6 \
  --n_real 50000 \
  --n_random 50000 \
  --deltas 0x0,0x1,0x2,0x10,0x100000001,0x10008,0x33,0xF0F
```

## Training models

Use the shared trainer in `Models/train.py`:

```bash
python3 Models/train.py \
  --data_path <dataset.npz> \
  --model {mlp|conv|resnet|logreg} \
  [--batch_size 128] \
  [--lr 1e-3] \
  [--epochs 20] \
  [--train_split 0.8] \
  [--split_seed 1234] \
  [--report_dir Reports]
```

Example:

```bash
python3 Models/train.py \
  --data_path gift64_R1_r5_s12345.npz \
  --model mlp
```

What the trainer does:

- loads `X` and `y` from the dataset
- splits the dataset into train/test partitions
- builds one of the models from `Models/models.py`
- trains with `BCEWithLogitsLoss` and Adam
- writes a text report with:
  - dataset metadata
  - train/test sizes
  - parameter count
  - best and final accuracy / advantage
  - confusion counts
  - epoch-by-epoch metrics

By default, training reports are written to `Reports/`.

## Row ablation analysis

`Other/rep_analysis_row_ablation.py` studies how performance changes when only a subset of rows from an `R2` dataset is used. It trains a logistic regression model on:

- the first 1 row
- the first 2 rows
- the first 4 rows
- all rows

It can also rank rows by a simple bias score computed from the training split.

Run it from the project root as a module:

```bash
python3 -m Other.rep_analysis_row_ablation \
  --npz_r2 <multi_diff_dataset.npz> \
  [--tag <name>] \
  [--split_seed 1234] \
  [--test_ratio 0.2] \
  [--batch_size 128] \
  [--lr 1e-3] \
  [--epochs 20] \
  [--report_dir reports] \
  [--use_top_rows_by_score]
```

Example:

```bash
python3 -m Other.rep_analysis_row_ablation \
  --npz_r2 rect64_R2_r6_s1521644627.npz \
  --tag rect_r6 \
  --use_top_rows_by_score
```

The ablation report includes:

- dataset metadata and shapes
- optional row scores
- accuracy / advantage for each row subset
- the best-performing subset of rows

By default, ablation reports are written to `reports/`.

## Model definitions

`Models/models.py` currently contains:

- `LogReg`: flatten all rows and bits, then apply a single linear layer
- `MLP`: two hidden fully connected layers
- `Conv1D`: 1D convolution over the 64-bit axis with rows as channels
- `ResNet1D`: residual 1D convolutional network with global average pooling

All models assume a block size of 64 bits and accept input tensors of shape `(batch, num_rows, 64)`.

## Practical notes

- The training script accepts any dataset whose `X` shape is `(N, num_rows, 64)`.
- `Models/train.py` assumes 64-bit blocks and will assert if the last dimension is not `64`.
- `Other/rep_analysis_row_ablation.py` is intended for `R2` datasets, not `R1`.
- The ablation script should be launched with `python3 -m Other.rep_analysis_row_ablation` from the repository root so that `Models` is importable.

## Minimal workflow

1. Generate an `R1` or `R2` dataset for a cipher and round count of interest.
2. Train one or more models with `Models/train.py`.
3. For multi-difference datasets, run `Other/rep_analysis_row_ablation.py` to compare row subsets.
4. Inspect the generated text reports to compare representation choices against architecture choices.
