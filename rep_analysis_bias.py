import numpy as np
import os
import matplotlib.pyplot as plt


def split_train_test(X, y, test_ratio=0.2, split_seed=123):
    rng = np.random.default_rng(split_seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def compute_bias_matrix(X, y):
    # X: (N, rows, 64), y: (N,)
    X_real = X[y == 1]
    X_rand = X[y == 0]
    p_real = X_real.mean(axis=0)  # (rows, 64)
    p_rand = X_rand.mean(axis=0)  # (rows, 64)
    B = np.abs(p_real - p_rand)  # (rows, 64)
    return B


def summarize_bias(B, topk=64, tau=0.01):
    flat = B.reshape(-1)
    flat_sorted = np.sort(flat)[::-1]
    max_bias = float(flat_sorted[0])
    topk_mean = float(flat_sorted[:topk].mean())
    count_tau = int((flat > tau).sum())
    return max_bias, topk_mean, count_tau, flat_sorted


def row_scores(B):
    # sum of biases per row
    return B.sum(axis=1)  # (rows,)


def save_sorted_bias_plot(sorted_bias_r1, sorted_bias_r2, out_pdf, title):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.figure()
    plt.plot(sorted_bias_r1, label="R1 (single-diff)")
    plt.plot(sorted_bias_r2, label="R2 (multi-diff)")
    plt.xlabel("Feature rank (sorted)")
    plt.ylabel("Per-bit bias |P(X=1|real) - P(X=1|rand)|")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def save_row_scores_table(
    scores, out_txt, caption="Row scores (sum of per-bit biases)"
):
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(caption + "\n")
        for i, s in enumerate(scores, start=1):
            f.write(f"row_{i}\t{s:.6f}\n")


def analyze_pair(
    npz_r1, npz_r2, figs_dir="figs", split_seed=123, topk=64, tau=0.01, tag="gift_r5"
):
    data1 = np.load(npz_r1, allow_pickle=False)
    data2 = np.load(npz_r2, allow_pickle=False)
    X1, y1 = data1["X"], data1["y"]
    X2, y2 = data2["X"], data2["y"]

    # split like your protocol (random 80/20)
    (_, _), (X1t, y1t) = split_train_test(X1, y1, test_ratio=0.2, split_seed=split_seed)
    (_, _), (X2t, y2t) = split_train_test(X2, y2, test_ratio=0.2, split_seed=split_seed)

    B1 = compute_bias_matrix(X1t, y1t)  # (1,64)
    B2 = compute_bias_matrix(X2t, y2t)  # (7,64)

    max1, topk1, cnt1, sorted1 = summarize_bias(B1, topk=topk, tau=tau)
    max2, topk2, cnt2, sorted2 = summarize_bias(B2, topk=topk, tau=tau)

    print(
        f"[{tag}] R1: max={max1:.6f}, top{topk}_mean={topk1:.6f}, count(b>{tau})={cnt1}"
    )
    print(
        f"[{tag}] R2: max={max2:.6f}, top{topk}_mean={topk2:.6f}, count(b>{tau})={cnt2}"
    )

    # plot
    out_plot = os.path.join(figs_dir, f"bias_sorted_{tag}.pdf")
    save_sorted_bias_plot(
        sorted1, sorted2, out_plot, title=f"Sorted per-bit biases ({tag})"
    )
    print("Saved:", out_plot)

    # row scores for R2
    rs = row_scores(B2)
    out_rows = os.path.join(figs_dir, f"row_scores_{tag}.txt")
    save_row_scores_table(
        rs, out_rows, caption=f"{tag}: Row scores S_i = sum_j b_(i,j)"
    )
    print("Saved:", out_rows)

    # return numbers for LaTeX
    return {
        "tag": tag,
        "R1": {"max": max1, "topk_mean": topk1, "count_tau": cnt1},
        "R2": {"max": max2, "topk_mean": topk2, "count_tau": cnt2},
        "row_scores_R2": rs,
        "plot": out_plot,
        "row_scores_file": out_rows,
    }


if __name__ == "__main__":
    # مثال: GIFT r=5
    analyze_pair(
        npz_r1="gift_r5_R1.npz",
        npz_r2="gift_r5_R2.npz",
        split_seed=123,
        topk=64,
        tau=0.01,
        tag="gift_r5",
    )

    # مثال: RECT r=6
    analyze_pair(
        npz_r1="rect_r6_R1.npz",
        npz_r2="rect_r6_R2.npz",
        split_seed=123,
        topk=64,
        tau=0.01,
        tag="rect_r6",
    )
