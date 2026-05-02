import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Models.models import LogReg

DEFAULT_TEST_RATIO = 0.2
DEFAULT_SPLIT_SEED = 1234
DEFAULT_REPORT_DIR = "Reports"
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 20


def split_train_test(X, y, test_ratio=0.2, split_seed=1234):
    rng = np.random.default_rng(split_seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def select_rows(X, rows_idx):
    return np.ascontiguousarray(X[:, rows_idx, :])


def load_npz_with_metadata(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.uint8)

    metadata = {}
    for key in data.files:
        if key not in ["X", "y"]:
            try:
                metadata[key] = data[key]
            except Exception:
                metadata[key] = "<unreadable>"

    return X, y, metadata


def format_metadata_value(value):
    try:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return str(value.item())
            if value.size == 1:
                return str(value.reshape(-1)[0])
            return str(value.tolist())
        return str(value)
    except Exception:
        return repr(value)


def make_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_tag(npz_r2, user_tag=None):
    if user_tag:
        return user_tag
    return Path(npz_r2).stem


def build_row_sets(num_rows, row_scores=None):
    base_sizes = [1, 2, 4, num_rows]
    sizes = []
    for k in base_sizes:
        k = min(k, num_rows)
        if k not in sizes:
            sizes.append(k)

    if row_scores is None:
        return {k: list(range(k)) for k in sizes}

    order = list(np.argsort(row_scores)[::-1])
    return {k: order[:k] for k in sizes}


def compute_row_scores_from_dataset(X, y):
    X_real = X[y == 1]
    X_rand = X[y == 0]
    p_real = X_real.mean(axis=0)
    p_rand = X_rand.mean(axis=0)
    B = np.abs(p_real - p_rand)
    return B.sum(axis=1)


def train_logreg_model(Xtr, ytr, num_rows, batch_size=128, lr=1e-3, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogReg(num_rows=num_rows).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    Xtr_tensor = torch.from_numpy(np.ascontiguousarray(Xtr.astype(np.float32)))
    ytr_tensor = torch.from_numpy(ytr.astype(np.float32))

    for _ in range(epochs):
        model.train()

        perm = torch.randperm(Xtr_tensor.size(0))
        for start in range(0, Xtr_tensor.size(0), batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_X = Xtr_tensor[batch_idx].contiguous().to(device)
            batch_y = ytr_tensor[batch_idx].to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model, device


def predict_logreg_model(model, device, Xte, batch_size=128):
    model.eval()
    Xte_tensor = torch.from_numpy(np.ascontiguousarray(Xte.astype(np.float32)))

    preds = []
    with torch.no_grad():
        for start in range(0, Xte_tensor.size(0), batch_size):
            batch_X = Xte_tensor[start : start + batch_size].contiguous().to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds.append((probs > 0.5).cpu().numpy().astype(np.uint8))

    return np.concatenate(preds, axis=0)


def evaluate_predictions(y_true, y_pred):
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    acc = float((y_true == y_pred).mean())
    adv = acc - 0.5

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "acc": float(acc),
        "adv": float(adv),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def write_report(report_path, tag, config, metadata, shapes, row_scores, results):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    best_k = max(results.keys(), key=lambda k: results[k]["acc"])
    best_result = results[best_k]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== ROW ABLATION REPORT ===\n\n")
        f.write(f"tag: {tag}\n\n")

        f.write("[CONFIG]\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[DATASET METADATA]\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"{k}: {format_metadata_value(v)}\n")
        else:
            f.write("No extra metadata found.\n")

        f.write("\n[DATA SHAPES]\n")
        for k, v in shapes.items():
            f.write(f"{k}: {v}\n")

        if row_scores is not None:
            f.write("\n[ROW SCORES USED FOR ORDERING]\n")
            for i, s in enumerate(row_scores, start=1):
                f.write(f"row_{i}: {float(s):.6f}\n")

        f.write("\n[SUMMARY]\n")
        f.write(f"best_k: {best_k}\n")
        f.write(f"best_acc: {best_result['acc']:.6f}\n")
        f.write(f"best_adv: {best_result['adv']:.6f}\n")
        f.write(f"best_rows: {best_result['rows']}\n")

        f.write("\n[RESULTS]\n")
        f.write("k\trows\tacc\tadv\tprecision\trecall\tspecificity\ttp\ttn\tfp\tfn\n")
        for k in sorted(results.keys()):
            r = results[k]
            f.write(
                f"{k}\t{r['rows']}\t{r['acc']:.6f}\t{r['adv']:.6f}\t"
                f"{r['precision']:.6f}\t{r['recall']:.6f}\t{r['specificity']:.6f}\t"
                f"{r['tp']}\t{r['tn']}\t{r['fp']}\t{r['fn']}\n"
            )


def run_ablation(
    npz_r2,
    split_seed=1234,
    test_ratio=0.2,
    use_top_rows_by_score=False,
    report_dir="reports",
    tag=None,
    batch_size=128,
    lr=1e-3,
    epochs=20,
):
    tag = infer_tag(npz_r2, tag)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)

    X, y, metadata = load_npz_with_metadata(npz_r2)

    (Xtr, ytr), (Xte, yte) = split_train_test(
        X, y, test_ratio=test_ratio, split_seed=split_seed
    )

    num_rows = X.shape[1]

    row_scores = None
    if use_top_rows_by_score:
        row_scores = compute_row_scores_from_dataset(Xtr, ytr)

    row_sets = build_row_sets(num_rows, row_scores=row_scores)

    results = {}
    for k, rows in row_sets.items():
        Xtr_rows = select_rows(Xtr, rows)
        Xte_rows = select_rows(Xte, rows)

        model, device = train_logreg_model(
            Xtr_rows, ytr, num_rows=k, batch_size=batch_size, lr=lr, epochs=epochs
        )
        pred = predict_logreg_model(model, device, Xte_rows, batch_size=batch_size)

        metrics = evaluate_predictions(yte, pred)
        metrics["rows"] = rows
        results[k] = metrics

        print(
            f"[{tag}] k={k} rows={rows} "
            f"acc={metrics['acc']:.6f} adv={metrics['adv']:.6f}"
        )

    report_path = os.path.join(report_dir, f"row_ablation_{tag}_{make_timestamp()}.txt")

    config = {
        "npz_r2": npz_r2,
        "split_seed": split_seed,
        "test_ratio": test_ratio,
        "use_top_rows_by_score": use_top_rows_by_score,
        "classifier": "Models.models.LogReg",
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
    }

    shapes = {
        "full_shape": tuple(X.shape),
        "train_shape": tuple(Xtr.shape),
        "test_shape": tuple(Xte.shape),
    }

    write_report(
        report_path=report_path,
        tag=tag,
        config=config,
        metadata=metadata,
        shapes=shapes,
        row_scores=row_scores,
        results=results,
    )

    print("Saved:", report_path)

    return {
        "tag": tag,
        "results": results,
        "row_scores": row_scores,
        "report_file": report_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Row ablation on R2 datasets")

    parser.add_argument("--npz_r2", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--report_dir", type=str, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--use_top_rows_by_score",
        action="store_true",
        help="Order rows by bias-based row scores computed on training split",
    )

    args = parser.parse_args()

    run_ablation(
        npz_r2=args.npz_r2,
        split_seed=args.split_seed,
        test_ratio=args.test_ratio,
        use_top_rows_by_score=args.use_top_rows_by_score,
        report_dir=args.report_dir,
        tag=args.tag,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()


# python row_ablation.py \
#   --npz_r2 rect64_R2_r6_s1521644627.npz \
#   --tag rect_r6 \
