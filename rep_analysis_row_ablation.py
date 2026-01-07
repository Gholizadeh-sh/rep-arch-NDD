import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def split_train_test(X, y, test_ratio=0.2, split_seed=123):
    rng = np.random.default_rng(split_seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def flatten_rows(X, rows_idx):
    # X: (N, 7, 64), rows_idx: list of row indices (0..6)
    Xs = X[:, rows_idx, :]  # (N, k, 64)
    return Xs.reshape(len(Xs), -1)  # (N, k*64)


def run_ablation(npz_r2, split_seed=123, use_top_rows_by_score=None):
    data = np.load(npz_r2, allow_pickle=False)
    X, y = data["X"], data["y"]  # X: (N,7,64)

    (Xtr, ytr), (Xte, yte) = split_train_test(
        X, y, test_ratio=0.2, split_seed=split_seed
    )

    row_sets = {
        1: [0],
        2: [0, 1],
        4: [0, 1, 2, 3],
        7: [0, 1, 2, 3, 4, 5, 6],
    }

    if use_top_rows_by_score is not None:
        order = list(np.argsort(use_top_rows_by_score)[::-1])  # descending
        row_sets = {
            1: order[:1],
            2: order[:2],
            4: order[:4],
            7: order[:7],
        }

    results = {}
    for k, rows in row_sets.items():
        Xtr_f = flatten_rows(Xtr, rows)
        Xte_f = flatten_rows(Xte, rows)

        clf = LogisticRegression(solver="lbfgs", max_iter=2000, n_jobs=-1)
        clf.fit(Xtr_f, ytr)
        pred = clf.predict(Xte_f)
        acc = accuracy_score(yte, pred)
        results[k] = acc
        print(f"k={k} rows={rows}  acc={acc:.6f}")

    return results


if __name__ == "__main__":
    run_ablation("gift_r5_R2.npz", split_seed=123)
    run_ablation("rect_r6_R2.npz", split_seed=123)
