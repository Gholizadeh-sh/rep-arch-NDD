# Code for `Rep vs. Arch in NDD`

This repository contains the reference implementation used in the paper **Representation vs. Architecture in Neural Differential Distinguishers: A Case Study on GIFT-64 and RECTANGLE-64**.

It includes two independent experiment folders:

- **`Gift-64/`** — dataset generation, cipher implementation, and training code for **GIFT-64**
- **`Rectangle-64/`** — dataset generation, cipher implementation, and training code for **RECTANGLE-64**

> If you use this code, please cite the paper and (optionally) this repository release.

---

## Requirements

- Python 3.9+ recommended
- Packages:
  - `numpy`
  - `torch`

Install dependencies:

```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quick start

### 1) Generate datasets

Datasets are saved as compressed NumPy files (`.npz`) containing:

- `X`: shape `(N, num_rows, 64)`
- `y`: shape `(N,)`

**GIFT-64**

```bash
cd Gift-64
python gift64_datasets.py
```

**RECTANGLE-64**

```bash
cd Rectangle-64
python make_rectangle_datasets.py
```

> The dataset scripts print the exact output filename(s) they saved (e.g., `gift64_...npz`, `rect64_...npz`).

### 2) Train a model

Each folder has its own training script. By default it loads a dataset file specified near the top of the script (`DATA_PATH`), and the model can be selected via `MODEL_TYPE`.

**GIFT-64**

```bash
cd Gift-64
# Edit DATA_PATH and MODEL_TYPE at the top if needed
python train_gift64.py
```

**RECTANGLE-64**

```bash
cd Rectangle-64
# Edit DATA_PATH and MODEL_TYPE at the top if needed
python train_rectangle-64.py
```

Models supported in the training scripts (set via the `MODEL_TYPE` constant):

- `logreg` (logistic regression baseline)
- `mlp`
- `conv`
- `resnet`

---

## Repository structure

```text
.
├── Gift-64/
│   ├── gift64_cipher.py             # GIFT-64 implementation
│   ├── gift64_datasets.py           # dataset generation (writes .npz)
│   ├── models.py                    # model definitions (MLP/Conv/ResNet/LogReg)
│   └── train_gift64.py              # training loop
├── Rectangle-64/
│   ├── rectangle64.py               # RECTANGLE-64 implementation (80-bit key)
│   ├── make_rectangle_datasets.py   # dataset generation (writes .npz)
│   ├── models.py                    # model definitions (MLP/Conv/ResNet/LogReg)
│   └── train_rectangle-64.py        # training loop
└── requirements.txt
```

> If your local filenames differ (e.g., `rectangle64_cipher.py` instead of `rectangle64.py`), update the import paths accordingly.

---

## Notes for reproducibility

- Dataset generation is randomized (seeds are set in the dataset scripts).
- Training uses `cuda` automatically if available; otherwise it runs on CPU.

### Paper-relevant notes (quick)

- The code supports two input representations used in the paper: **R1 (single-difference)** and **R2 (multi-difference matrix)**.
- The **difference values and bit ordering (LSB → MSB)** follow the paper and are implemented directly in the dataset scripts.
- Many experiments in the paper vary **seed/key/rounds**; these are intentionally kept as editable constants at the top of the scripts.

If you want _exactly reproducible_ training results across machines, consider also fixing:

- torch seed, numpy seed
- deterministic CUDA settings (optional)

---

## Citation

**Paper**

```bibtex
@article{YOURKEY,
  title   = {Representation vs. Architecture in Neural Differential Distinguishers: A Case Study on GIFT-64 and RECTANGLE-64},
  author  = {Alireza Gholizadeh Shahrbejari and Reza Ebrahimi Atani},
  journal = {<VENUE>},
  year    = {2025}
}
```

**Repository (optional)**
Add a GitHub release (e.g., `v1.0.0`) and then cite that release/tag in your paper.

---

## License

Add a `LICENSE` file (MIT/BSD/Apache-2.0 are common for research code).
If you used third-party code, include their license notices as required.

```

```
