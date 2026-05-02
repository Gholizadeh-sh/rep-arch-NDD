import numpy as np
import random
import secrets
from gift64_cipher import gift64_encrypt


# =========================
# FAST BIT CONVERSION
# =========================


def ct_to_bits(ct: int) -> np.ndarray:
    return np.unpackbits(np.array([ct], dtype=">u8").view(np.uint8))[::-1].astype(
        np.uint8
    )


# =========================
# SHUFFLE
# =========================


def shuffle_dataset(X: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# =========================
# MULTI DIFF
# =========================


def make_sample_real_multidiff(
    deltas,
    key,
    r_start,
    r_end,
    use_differences=True,
    include_base_row=False,
    plaintext=None,
):
    base_plaintext = plaintext if plaintext is not None else random.getrandbits(64)

    ct0 = gift64_encrypt(base_plaintext ^ deltas[0], key, r_start, r_end)

    rows = []

    for j, delta in enumerate(deltas):
        ct = gift64_encrypt(base_plaintext ^ delta, key, r_start, r_end)

        if use_differences:
            bits = ct_to_bits(ct ^ ct0)
        else:
            bits = ct_to_bits(ct)

        if (not include_base_row) and j == 0:
            continue

        rows.append(bits)

    return np.stack(rows)


def make_sample_random_multidiff(
    deltas,
    key,
    r_start,
    r_end,
    use_differences=True,
    include_base_row=False,
):
    ct0 = gift64_encrypt(random.getrandbits(64), key, r_start, r_end)

    rows = []

    for j, _ in enumerate(deltas):
        ct = gift64_encrypt(random.getrandbits(64), key, r_start, r_end)

        if use_differences:
            bits = ct_to_bits(ct ^ ct0)
        else:
            bits = ct_to_bits(ct)

        if (not include_base_row) and j == 0:
            continue

        rows.append(bits)

    return np.stack(rows)


def generate_binary_dataset_multidiff(
    n_real,
    n_random,
    deltas,
    key,
    r_start,
    r_end,
    use_differences=True,
    include_base_row=False,
    seed=42,
    plaintexts=None,
):
    random.seed(seed)
    np.random.seed(seed)

    probe = make_sample_real_multidiff(
        deltas, key, r_start, r_end, use_differences, include_base_row
    )

    rows, bits = probe.shape

    X = np.zeros((n_real + n_random, rows, bits), dtype=np.uint8)
    y = np.zeros(n_real + n_random, dtype=np.uint8)

    idx = 0

    for i in range(n_real):
        pt = plaintexts[i] if plaintexts is not None else None

        X[idx] = make_sample_real_multidiff(
            deltas,
            key,
            r_start,
            r_end,
            use_differences,
            include_base_row,
            pt,
        )
        y[idx] = 1
        idx += 1

    for _ in range(n_random):
        X[idx] = make_sample_random_multidiff(
            deltas,
            key,
            r_start,
            r_end,
            use_differences,
            include_base_row,
        )
        idx += 1

    return shuffle_dataset(X, y, seed)


# =========================
# SINGLE DIFF
# =========================


def make_sample_real_single_diff(
    delta,
    key,
    r_start,
    r_end,
    plaintext=None,
):
    P = plaintext if plaintext is not None else random.getrandbits(64)

    C = gift64_encrypt(P, key, r_start, r_end)
    C2 = gift64_encrypt(P ^ delta, key, r_start, r_end)

    return ct_to_bits(C ^ C2).reshape(1, 64)


def make_sample_random_single_diff(key, r_start, r_end):
    C = gift64_encrypt(random.getrandbits(64), key, r_start, r_end)
    C2 = gift64_encrypt(random.getrandbits(64), key, r_start, r_end)

    return ct_to_bits(C ^ C2).reshape(1, 64)


def generate_binary_dataset_single_diff(
    n_real,
    n_random,
    delta,
    key,
    r_start,
    r_end,
    seed=42,
    plaintexts=None,
):
    random.seed(seed)
    np.random.seed(seed)

    X = np.zeros((n_real + n_random, 1, 64), dtype=np.uint8)
    y = np.zeros(n_real + n_random, dtype=np.uint8)

    idx = 0

    for i in range(n_real):
        pt = plaintexts[i] if plaintexts is not None else None

        X[idx] = make_sample_real_single_diff(
            delta,
            key,
            r_start,
            r_end,
            pt,
        )
        y[idx] = 1
        idx += 1

    for _ in range(n_random):
        X[idx] = make_sample_random_single_diff(
            key,
            r_start,
            r_end,
        )
        idx += 1

    return shuffle_dataset(X, y, seed)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["single", "multi", "both"], required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--n_real", type=int, required=True)
    parser.add_argument("--n_random", type=int, required=True)

    parser.add_argument("--delta", type=str)
    parser.add_argument("--deltas", type=str)

    parser.add_argument("--plaintext_file", default="plaintexts64.txt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--key", type=str)

    args = parser.parse_args()

    r_start = 0
    r_end = args.rounds

    # Per execution: use provided seed/key if given, otherwise generate fresh ones
    run_seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
    master_key = int(args.key, 0) if args.key is not None else secrets.randbits(128)

    print("GIFT-64 dataset generator")
    print("run_seed =", run_seed)
    print("master_key =", hex(master_key))

    # load plaintexts
    plaintexts = []

    try:
        with open(args.plaintext_file) as f:
            for line in f:
                plaintexts.append(int(line.strip(), 16))

        if len(plaintexts) < args.n_real:
            raise ValueError("Not enough plaintexts in file")

        print("plaintexts loaded:", len(plaintexts))

    except FileNotFoundError:
        plaintexts = None
        print("plaintext file not found -> random plaintexts")

    if args.mode in ["single", "both"]:
        delta = int(args.delta, 16)

        X, y = generate_binary_dataset_single_diff(
            args.n_real,
            args.n_random,
            delta,
            master_key,
            r_start,
            r_end,
            run_seed,
            plaintexts,
        )

        fname = f"gift64_R1_r{args.rounds}_s{run_seed}.npz"

        np.savez_compressed(
            fname,
            X=X,
            y=y,
            key=np.array([master_key], dtype=object),
            seed=np.array([run_seed], dtype=np.uint32),
            rounds=np.array([args.rounds], dtype=np.uint16),
            delta=np.array([delta], dtype=object),
        )

        print("saved", fname, "shape", X.shape)

    if args.mode in ["multi", "both"]:
        deltas = [int(x, 16) for x in args.deltas.split(",")]

        X, y = generate_binary_dataset_multidiff(
            args.n_real,
            args.n_random,
            deltas,
            master_key,
            r_start,
            r_end,
            True,
            False,
            run_seed,
            plaintexts,
        )

        fname = f"gift64_R2_r{args.rounds}_s{run_seed}.npz"

        np.savez_compressed(
            fname,
            X=X,
            y=y,
            key=np.array([master_key], dtype=object),
            seed=np.array([run_seed], dtype=np.uint32),
            rounds=np.array([args.rounds], dtype=np.uint16),
            deltas=np.array(deltas, dtype=object),
            use_differences=np.array([True], dtype=np.uint8),
            include_base_row=np.array([False], dtype=np.uint8),
        )

        print("saved", fname, "shape", X.shape)

# python make_rectangle_datasets.py \
# --mode single \
# --rounds 6 \
# --n_real 50000 \
# --n_random 50000 \
# --delta 0x1


# python make_rectangle_datasets.py \
# --mode multi \
# --rounds 6 \
# --n_real 50000 \
# --n_random 50000 \
# --deltas 0x0,0x1,0x2,0x10,0x100000001,0x10008,0x33,0xF0F


# python dataset_generator.py \
# --mode both \
# --rounds 5 \
# --n_real 32768 \
# --n_random 32768 \
# --delta 0x1 \
# --deltas 0x0,0x1,0x2,0x10,0x100000001,0x10008,0x33,0xF0F
