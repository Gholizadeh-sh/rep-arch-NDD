# gift64_datasets.py

import numpy as np
import random
from typing import List, Tuple
from gift64_cipher import gift64_encrypt


def ct_to_bits(ct: int, n_bits: int = 64) -> np.ndarray:
    """Convert integer ciphertext to bit vector (LSB first)."""
    return np.array([(ct >> i) & 1 for i in range(n_bits)], dtype=np.uint8)


# =========================
#   R2: MULTI-DIFF MATRIX
# =========================


def make_sample_real_multidiff(
    deltas: List[int],
    key: int,
    r_start: int,
    r_end: int,
    use_differences: bool = True,
    include_base_row: bool = False,
) -> np.ndarray:
    """
    Real sample for multi-diff:
      - base P
      - for each delta: P^delta → C
    If use_differences=True:
      rows = C_i XOR C_0
    else:
      rows = C_i

    Returns: matrix (num_rows, 64)
    """
    base_plaintext = random.getrandbits(64)

    # Assume deltas[0] == 0 for building C0
    delta0 = deltas[0]
    pt0 = base_plaintext ^ delta0
    ct0 = gift64_encrypt(pt0, key, r_start, r_end)

    rows = []
    for j, delta in enumerate(deltas):
        pt = base_plaintext ^ delta
        ct = gift64_encrypt(pt, key, r_start, r_end)

        if use_differences:
            diff = ct ^ ct0
            bits = ct_to_bits(diff)
        else:
            bits = ct_to_bits(ct)

        if (not include_base_row) and (j == 0):
            continue
        rows.append(bits)

    return np.stack(rows, axis=0)  # (num_rows, 64)


def make_sample_random_multidiff(
    deltas: List[int],
    key: int,
    r_start: int,
    r_end: int,
    use_differences: bool = True,
    include_base_row: bool = False,
) -> np.ndarray:
    """
    Random sample for multi-diff:
      - each row: independent random plaintext
    If use_differences=True:
      rows = C_i XOR C_0, where C_0 from independent random P0.
    """
    pt0 = random.getrandbits(64)
    ct0 = gift64_encrypt(pt0, key, r_start, r_end)

    rows = []
    for j, _delta in enumerate(deltas):
        pt = random.getrandbits(64)
        ct = gift64_encrypt(pt, key, r_start, r_end)

        if use_differences:
            diff = ct ^ ct0
            bits = ct_to_bits(diff)
        else:
            bits = ct_to_bits(ct)

        if (not include_base_row) and (j == 0):
            continue
        rows.append(bits)

    return np.stack(rows, axis=0)


def generate_binary_dataset_multidiff(
    n_real: int,
    n_random: int,
    deltas: List[int],
    key: int,
    r_start: int,
    r_end: int,
    use_differences: bool = True,
    include_base_row: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset for R2 (multi-diff matrix).
      X: (N, num_rows, 64)
      y: (N,)
    """
    random.seed(seed)
    np.random.seed(seed)

    # probe shape
    sample = make_sample_real_multidiff(
        deltas,
        key,
        r_start,
        r_end,
        use_differences=use_differences,
        include_base_row=include_base_row,
    )
    num_rows, n_bits = sample.shape

    X = np.zeros((n_real + n_random, num_rows, n_bits), dtype=np.uint8)
    y = np.zeros((n_real + n_random,), dtype=np.uint8)

    idx = 0
    for _ in range(n_real):
        X[idx] = make_sample_real_multidiff(
            deltas,
            key,
            r_start,
            r_end,
            use_differences=use_differences,
            include_base_row=include_base_row,
        )
        y[idx] = 1
        idx += 1

    for _ in range(n_random):
        X[idx] = make_sample_random_multidiff(
            deltas,
            key,
            r_start,
            r_end,
            use_differences=use_differences,
            include_base_row=include_base_row,
        )
        y[idx] = 0
        idx += 1

    return X, y


# =========================
#   R1: SINGLE-DIFF
# =========================


def make_sample_real_single_diff(
    delta: int, key: int, r_start: int, r_end: int
) -> np.ndarray:
    """
    Real sample for single-diff:
      - P random
      - P' = P XOR delta
      - C, C' = E(P), E(P')
      - row = bits of C XOR C'
    Returns: matrix (1, 64)
    """
    P = random.getrandbits(64)
    P2 = P ^ delta
    C = gift64_encrypt(P, key, r_start, r_end)
    C2 = gift64_encrypt(P2, key, r_start, r_end)
    diff = C ^ C2
    bits = ct_to_bits(diff)
    return bits.reshape(1, -1)  # (1, 64)


def make_sample_random_single_diff(key: int, r_start: int, r_end: int) -> np.ndarray:
    """
    Random sample for single-diff:
      - P, Q independent random
      - C, C' = E(P), E(Q)
      - row = bits of C XOR C'
    """
    P = random.getrandbits(64)
    Q = random.getrandbits(64)
    C = gift64_encrypt(P, key, r_start, r_end)
    C2 = gift64_encrypt(Q, key, r_start, r_end)
    diff = C ^ C2
    bits = ct_to_bits(diff)
    return bits.reshape(1, -1)  # (1, 64)


def generate_binary_dataset_single_diff(
    n_real: int,
    n_random: int,
    delta: int,
    key: int,
    r_start: int,
    r_end: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset for R1 (single-diff, C XOR C'):
      X: (N, 1, 64)
      y: (N,)
    """
    random.seed(seed)
    np.random.seed(seed)

    X = np.zeros((n_real + n_random, 1, 64), dtype=np.uint8)
    y = np.zeros((n_real + n_random,), dtype=np.uint8)

    idx = 0
    for _ in range(n_real):
        X[idx] = make_sample_real_single_diff(delta, key, r_start, r_end)
        y[idx] = 1
        idx += 1

    for _ in range(n_random):
        X[idx] = make_sample_random_single_diff(key, r_start, r_end)
        y[idx] = 0
        idx += 1

    return X, y


# =========================
#   MAIN: EXAMPLE BUILDER
# =========================

if __name__ == "__main__":
    from gift64_cipher import (
        gift64_encrypt,
    )  # just to ensure import works

    rounds = 5
    r_start = 0
    r_end = r_start + rounds

    n_real = 2**13
    n_random = 2**13

    seed = 320
    random.seed(seed)
    master_key = random.getrandbits(128)

    # ---------- R2: multi-diff diff-matrix ----------
    deltas = [
        0x0000000000000000,  # delta0
        0x0000000000000001,
        0x0000000000000002,
        0x0000000000000010,
        0x0000000100000001,
        0x0000000000010008,
        0x0000000000000033,
        0x0000000000000F0F,
    ]

    X2, y2 = generate_binary_dataset_multidiff(
        n_real=n_real,
        n_random=n_random,
        deltas=deltas,
        key=master_key,
        r_start=r_start,
        r_end=r_end,
        use_differences=True,
        include_base_row=False,
        seed=seed,
    )

    out_path_R2 = f"gift64_R2_multidiff_r{rounds}_n{n_real+n_random}_seed{seed}.npz"
    np.savez_compressed(
        out_path_R2,
        X=X2,
        y=y2,
        rounds=rounds,
        mode="R2_multidiff",
        deltas=np.array(deltas, dtype=np.uint64),
        seed=seed,
    )
    print("Saved R2 dataset to:", out_path_R2)
    print("X2 shape:", X2.shape, "y2 shape:", y2.shape)

    # ---------- R1: single-diff ----------
    delta_single = 0x0000000000000001

    X1, y1 = generate_binary_dataset_single_diff(
        n_real=n_real,
        n_random=n_random,
        delta=delta_single,
        key=master_key,
        r_start=r_start,
        r_end=r_end,
        seed=seed,
    )

    out_path_R1 = f"gift64_R1_single_r{rounds}_n{n_real+n_random}_seed{seed}.npz"
    np.savez_compressed(
        out_path_R1,
        X=X1,
        y=y1,
        rounds=rounds,
        mode="R1_single",
        delta=np.uint64(delta_single),
        seed=seed,
    )
    print("Saved R1 dataset to:", out_path_R1)
    print("X1 shape:", X1.shape, "y1 shape:", y1.shape)
