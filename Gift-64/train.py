# train_gift64.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from models import GiftMLP, GiftConv1D, GiftResNet1D, LogReg

# =========================
# Settings
# =========================

# Path to dataset
DATA_PATH = "gift64_R2_multidiff_r5_n65536_seed42.npz"
MODEL_TYPE = "logreg"  # "mlp", "conv", "resnet", "logreg"

BATCH_SIZE = 128
LR = 1e-3
N_EPOCHS = 20
TRAIN_SPLIT = 0.8


def build_model(model_type: str, num_rows: int) -> nn.Module:
    if model_type == "mlp":
        return GiftMLP(num_rows=num_rows)
    elif model_type == "conv":
        return GiftConv1D(num_rows=num_rows)
    elif model_type == "resnet":
        return GiftResNet1D(num_rows=num_rows)
    elif model_type == "logreg":
        return LogReg(num_rows)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


def main():
    # -------------------------
    # 1) Load dataset
    # -------------------------
    data = np.load(DATA_PATH)
    X = data["X"]  # (N, num_rows, 64)
    y = data["y"]  # (N,)

    print("Loaded X shape:", X.shape)
    print("Loaded y shape:", y.shape)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    N, num_rows, n_bits = X.shape
    assert n_bits == 64, "This training script assumes block size = 64."

    X_tensor = torch.from_numpy(X)  # (N, num_rows, 64)
    y_tensor = torch.from_numpy(y)  # (N,)

    dataset = TensorDataset(X_tensor, y_tensor)

    # train/test split
    n_train = int(TRAIN_SPLIT * N)
    n_test = N - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # 2) Build model
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model type:", MODEL_TYPE)

    model = build_model(MODEL_TYPE, num_rows=num_rows).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------------------------
    # 3) Training loop
    # -------------------------
    for epoch in range(1, N_EPOCHS + 1):
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

        # eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_X)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_X.size(0)

        test_acc = test_correct / test_total

        print(
            f"Epoch {epoch:2d}/{N_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
