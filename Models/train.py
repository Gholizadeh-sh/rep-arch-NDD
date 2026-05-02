import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from models import MLP, Conv1D, ResNet1D, LogReg


DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 20
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_SPLIT_SEED = 1234
DEFAULT_REPORT_DIR = "Reports"


def build_model(model_type: str, num_rows: int) -> nn.Module:
    model_type = model_type.lower()

    if model_type == "mlp":
        return MLP(num_rows=num_rows)
    if model_type == "conv":
        return Conv1D(num_rows=num_rows)
    if model_type == "resnet":
        return ResNet1D(num_rows=num_rows)
    if model_type == "logreg":
        return LogReg(num_rows=num_rows)

    raise ValueError(f"Unknown model type: {model_type}")


def load_dataset(data_path: str):
    data = np.load(data_path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

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


def compute_class_counts(y: np.ndarray):
    y_int = y.astype(np.uint8)
    n_total = len(y_int)
    n_pos = int((y_int == 1).sum())
    n_neg = int((y_int == 0).sum())
    return n_total, n_pos, n_neg


def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_loss += loss.item() * batch_X.size(0)
            correct += (preds == batch_y).sum().item()
            total += batch_X.size(0)

            tp += ((preds == 1) & (batch_y == 1)).sum().item()
            tn += ((preds == 0) & (batch_y == 0)).sum().item()
            fp += ((preds == 1) & (batch_y == 0)).sum().item()
            fn += ((preds == 0) & (batch_y == 1)).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    adv = acc - 0.5

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "loss": avg_loss,
        "acc": acc,
        "adv": adv,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_report_path(report_dir: str, data_path: str, model_type: str) -> str:
    os.makedirs(report_dir, exist_ok=True)

    dataset_stem = Path(data_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(report_dir) / f"{dataset_stem}_{model_type}_{timestamp}.txt")


def write_report(
    report_path: str,
    config: dict,
    metadata: dict,
    train_size: int,
    test_size: int,
    num_rows: int,
    n_bits: int,
    n_params: int,
    history: list,
):
    best_epoch_record = max(history, key=lambda x: x["test_acc"])
    last_record = history[-1]
    min_test_loss_epoch = min(history, key=lambda x: x["test_loss"])
    max_overfit_gap = max(x["train_acc"] - x["test_acc"] for x in history)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== TRAINING REPORT ===\n\n")

        f.write("[CONFIG]\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[DATASET METADATA]\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"{k}: {format_metadata_value(v)}\n")
        else:
            f.write("No extra metadata found in dataset.\n")

        f.write("\n[DATA SHAPES]\n")
        f.write(f"train_size: {train_size}\n")
        f.write(f"test_size: {test_size}\n")
        f.write(f"num_rows: {num_rows}\n")
        f.write(f"block_size: {n_bits}\n")
        f.write(f"trainable_parameters: {n_params}\n")

        f.write("\n[SUMMARY]\n")
        f.write(f"best_test_acc: {best_epoch_record['test_acc']:.6f}\n")
        f.write(f"best_test_adv: {best_epoch_record['test_adv']:.6f}\n")
        f.write(f"best_epoch: {best_epoch_record['epoch']}\n")
        f.write(f"best_epoch_train_acc: {best_epoch_record['train_acc']:.6f}\n")
        f.write(f"best_epoch_train_loss: {best_epoch_record['train_loss']:.6f}\n")
        f.write(f"best_epoch_test_loss: {best_epoch_record['test_loss']:.6f}\n")
        f.write(f"final_train_acc: {last_record['train_acc']:.6f}\n")
        f.write(f"final_test_acc: {last_record['test_acc']:.6f}\n")
        f.write(f"final_train_loss: {last_record['train_loss']:.6f}\n")
        f.write(f"final_test_loss: {last_record['test_loss']:.6f}\n")
        f.write(f"final_test_adv: {last_record['test_adv']:.6f}\n")
        f.write(f"max_overfit_gap_train_minus_test_acc: {max_overfit_gap:.6f}\n")
        f.write(f"epoch_min_test_loss: {min_test_loss_epoch['epoch']}\n")
        f.write(f"min_test_loss: {min_test_loss_epoch['test_loss']:.6f}\n")

        f.write("\n[FINAL CONFUSION COUNTS]\n")
        f.write(f"TP: {last_record['test_tp']}\n")
        f.write(f"TN: {last_record['test_tn']}\n")
        f.write(f"FP: {last_record['test_fp']}\n")
        f.write(f"FN: {last_record['test_fn']}\n")
        f.write(f"precision: {last_record['test_precision']:.6f}\n")
        f.write(f"recall: {last_record['test_recall']:.6f}\n")
        f.write(f"specificity: {last_record['test_specificity']:.6f}\n")

        f.write("\n[EPOCH-BY-EPOCH]\n")
        f.write(
            "epoch\ttrain_loss\ttrain_acc\ttrain_adv\t"
            "test_loss\ttest_acc\ttest_adv\t"
            "test_precision\ttest_recall\ttest_specificity\n"
        )

        for row in history:
            f.write(
                f"{row['epoch']}\t"
                f"{row['train_loss']:.6f}\t"
                f"{row['train_acc']:.6f}\t"
                f"{row['train_adv']:.6f}\t"
                f"{row['test_loss']:.6f}\t"
                f"{row['test_acc']:.6f}\t"
                f"{row['test_adv']:.6f}\t"
                f"{row['test_precision']:.6f}\t"
                f"{row['test_recall']:.6f}\t"
                f"{row['test_specificity']:.6f}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generic trainer for 64-bit cipher datasets"
    )

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp", "conv", "resnet", "logreg"],
    )

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train_split", type=float, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--report_dir", type=str, default=DEFAULT_REPORT_DIR)

    args = parser.parse_args()

    X, y, metadata = load_dataset(args.data_path)

    print("Loaded X shape:", X.shape)
    print("Loaded y shape:", y.shape)

    N, num_rows, n_bits = X.shape
    assert n_bits == 64, "This training script assumes block size = 64."

    n_total, n_pos, n_neg = compute_class_counts(y)
    print(f"Dataset size: {n_total} | positives: {n_pos} | negatives: {n_neg}")

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(X_tensor, y_tensor)

    n_train = int(args.train_split * N)
    n_test = N - n_train

    split_generator = torch.Generator().manual_seed(args.split_seed)
    train_set, test_set = random_split(
        dataset, [n_train, n_test], generator=split_generator
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model type:", args.model)

    model = build_model(args.model, num_rows=num_rows).to(device)
    n_params = count_parameters(model)
    print("Trainable parameters:", n_params)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_X.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        train_adv = train_acc - 0.5

        test_metrics = evaluate_model(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_adv": train_adv,
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_adv": test_metrics["adv"],
            "test_tp": test_metrics["tp"],
            "test_tn": test_metrics["tn"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_specificity": test_metrics["specificity"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['acc']:.4f} | "
            f"test_adv={test_metrics['adv']:.4f}"
        )

    report_path = make_report_path(args.report_dir, args.data_path, args.model)

    config = {
        "data_path": args.data_path,
        "model": args.model,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_split": args.train_split,
        "split_seed": args.split_seed,
        "device": str(device),
    }

    write_report(
        report_path=report_path,
        config=config,
        metadata=metadata,
        train_size=n_train,
        test_size=n_test,
        num_rows=num_rows,
        n_bits=n_bits,
        n_params=n_params,
        history=history,
    )

    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()


# python train.py --data_path gift64_R1_r5_s12345.npz --model mlp
