# make_rectangle_datasets.py
import numpy as np
import random
from rectangle64_cipher import encrypt_rect64_80

# =========================
# Settings
# =========================

N_SAMPLES = 2**16  # N = N_real + N_rand
ROUNDS_LIST = [6]

# KEY SETTINGS
# MASTER_KEY_80 = random.getrandbits(80) # Random 80-bit master key
MASTER_KEY_80 = 0x964135ABEC39742AA62E  # Fixed master key

GLOBAL_SEED = 777

# Single delta for R1
DELTA_R1 = 0x0000000000000001

# Multiple deeltas for R2
DELTAS_R2 = [
    0x0000000000000000,
    0x0000000000000001,
    0x0000000000000002,
    0x0000000000000010,
    0x0000000100000001,
    0x0000000000010008,
    0x0000000000000033,
    0x0000000000000F0F,
]

# =========================
# Helper functions
# =========================


def int_to_bits(x: int, n_bits: int = 64):
    return [(x >> i) & 1 for i in range(n_bits)]


def make_dataset_R1_single(num_samples: int, rounds: int, delta: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    n_real = num_samples // 2
    n_rand = num_samples - n_real

    X = []
    y = []

    # --- real samples (label = 1)
    for _ in range(n_real):
        P = random.getrandbits(64)
        Pp = P ^ delta
        C = encrypt_rect64_80(P, MASTER_KEY_80, rounds)
        Cp = encrypt_rect64_80(Pp, MASTER_KEY_80, rounds)
        diff = C ^ Cp
        bits = int_to_bits(diff, 64)
        X.append([bits])
        y.append(1)

    # --- random samples (label = 0)
    for _ in range(n_rand):
        P = random.getrandbits(64)
        Q = random.getrandbits(64)
        C = encrypt_rect64_80(P, MASTER_KEY_80, rounds)
        Cp = encrypt_rect64_80(Q, MASTER_KEY_80, rounds)
        diff = C ^ Cp
        bits = int_to_bits(diff, 64)
        X.append([bits])
        y.append(0)

    X = np.array(X, dtype=np.uint8)  # (N, 1, 64)
    y = np.array(y, dtype=np.uint8)

    # shuffle
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X, y


def make_dataset_R2_multidiff(num_samples: int, rounds: int, deltas, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    d = len(deltas)
    n_rows = d - 1
    n_real = num_samples // 2
    n_rand = num_samples - n_real

    X = []
    y = []

    # --- real samples (label = 1)
    for _ in range(n_real):
        P0 = random.getrandbits(64)
        C_list = []
        for delta in deltas:
            Pi = P0 ^ delta
            Ci = encrypt_rect64_80(Pi, MASTER_KEY_80, rounds)
            C_list.append(Ci)
        C0 = C_list[0]
        rows = []
        for i in range(1, d):
            Di = C_list[i] ^ C0
            rows.append(int_to_bits(Di, 64))
        X.append(rows)
        y.append(1)

    # --- random samples (label = 0)
    for _ in range(n_rand):
        C_list = []
        for _delta in deltas:
            Pi = random.getrandbits(64)
            Ci = encrypt_rect64_80(Pi, MASTER_KEY_80, rounds)
            C_list.append(Ci)
        C0 = C_list[0]
        rows = []
        for i in range(1, d):
            Di = C_list[i] ^ C0
            rows.append(int_to_bits(Di, 64))
        X.append(rows)
        y.append(0)

    X = np.array(X, dtype=np.uint8)  # (N, n_rows, 64)
    y = np.array(y, dtype=np.uint8)

    # shuffle
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X, y


# =========================
#  main
# =========================


def main():
    print("Master key (80-bit) used for all RECTANGLE datasets:")
    print(f"{MASTER_KEY_80:020X}")

    for r in ROUNDS_LIST:
        print(f"\n=== Generating datasets for RECTANGLE-80, r = {r} rounds ===")

        # ----- R1 -----
        X1, y1 = make_dataset_R1_single(
            num_samples=N_SAMPLES,
            rounds=r,
            delta=DELTA_R1,
            seed=GLOBAL_SEED + r + 100,
        )
        fname1 = f"rect64_R1_single_r{r}_n{N_SAMPLES}_seed{GLOBAL_SEED}.npz"
        np.savez_compressed(fname1, X=X1, y=y1)
        print(
            f"Saved R1 dataset to {fname1} with X shape={X1.shape}, y shape={y1.shape}"
        )

        # ----- R2 -----
        X2, y2 = make_dataset_R2_multidiff(
            num_samples=N_SAMPLES,
            rounds=r,
            deltas=DELTAS_R2,
            seed=GLOBAL_SEED + r + 200,
        )
        fname2 = f"rect64_R2_multidiff_r{r}_n{N_SAMPLES}_seed{GLOBAL_SEED}.npz"
        np.savez_compressed(fname2, X=X2, y=y2)
        print(
            f"Saved R2 dataset to {fname2} with X shape={X2.shape}, y shape={y2.shape}"
        )


if __name__ == "__main__":
    main()
