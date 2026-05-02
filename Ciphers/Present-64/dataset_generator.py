import numpy as np
import random
import secrets
from present64_cipher import encrypt_present64_80


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
# SINGLE DIFFERENTIAL
# =========================


def make_sample_real_single(delta, key, rounds, plaintext=None):
    P = plaintext if plaintext is not None else random.getrandbits(64)

    C = encrypt_present64_80(P, key, rounds)
    Cp = encrypt_present64_80(P ^ delta, key, rounds)

    return ct_to_bits(C ^ Cp).reshape(1, 64)


def make_sample_random_single(key, rounds):
    C = encrypt_present64_80(random.getrandbits(64), key, rounds)
    Cp = encrypt_present64_80(random.getrandbits(64), key, rounds)

    return ct_to_bits(C ^ Cp).reshape(1, 64)


def generate_dataset_single(
    n_real,
    n_random,
    delta,
    key,
    rounds,
    seed=0,
    plaintexts=None,
):
    random.seed(seed)
    np.random.seed(seed)

    X = np.zeros((n_real + n_random, 1, 64), dtype=np.uint8)
    y = np.zeros(n_real + n_random, dtype=np.uint8)

    idx = 0

    for i in range(n_real):
        pt = plaintexts[i] if plaintexts is not None else None

        X[idx] = make_sample_real_single(
            delta,
            key,
            rounds,
            pt,
        )

        y[idx] = 1
        idx += 1

    for _ in range(n_random):
        X[idx] = make_sample_random_single(
            key,
            rounds,
        )
        idx += 1

    return shuffle_dataset(X, y, seed)


# =========================
# MULTI DIFFERENTIAL
# =========================


def make_sample_real_multi(
    deltas,
    key,
    rounds,
    plaintext=None,
):
    P0 = plaintext if plaintext is not None else random.getrandbits(64)

    ct_list = []

    for d in deltas:
        Pi = P0 ^ d
        Ci = encrypt_present64_80(Pi, key, rounds)
        ct_list.append(Ci)

    C0 = ct_list[0]

    rows = []

    for i in range(1, len(deltas)):
        rows.append(ct_to_bits(ct_list[i] ^ C0))

    return np.stack(rows)


def make_sample_random_multi(
    deltas,
    key,
    rounds,
):
    ct_list = []

    for _ in deltas:
        Pi = random.getrandbits(64)
        Ci = encrypt_present64_80(Pi, key, rounds)
        ct_list.append(Ci)

    C0 = ct_list[0]

    rows = []

    for i in range(1, len(deltas)):
        rows.append(ct_to_bits(ct_list[i] ^ C0))

    return np.stack(rows)


def generate_dataset_multi(
    n_real,
    n_random,
    deltas,
    key,
    rounds,
    seed=0,
    plaintexts=None,
):
    random.seed(seed)
    np.random.seed(seed)

    probe = make_sample_real_multi(
        deltas,
        key,
        rounds,
    )

    rows, bits = probe.shape

    X = np.zeros((n_real + n_random, rows, bits), dtype=np.uint8)
    y = np.zeros(n_real + n_random, dtype=np.uint8)

    idx = 0

    for i in range(n_real):
        pt = plaintexts[i] if plaintexts is not None else None

        X[idx] = make_sample_real_multi(
            deltas,
            key,
            rounds,
            pt,
        )

        y[idx] = 1
        idx += 1

    for _ in range(n_random):
        X[idx] = make_sample_random_multi(
            deltas,
            key,
            rounds,
        )
        idx += 1

    return shuffle_dataset(X, y, seed)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["single", "multi", "both"],
        required=True,
    )

    parser.add_argument("--rounds", type=int, required=True)

    parser.add_argument("--n_real", type=int, required=True)
    parser.add_argument("--n_random", type=int, required=True)

    parser.add_argument("--delta", type=str)
    parser.add_argument("--deltas", type=str)

    parser.add_argument(
        "--plaintext_file",
        default="plaintexts64.txt",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--key", type=str)

    args = parser.parse_args()

    rounds = args.rounds

    # Per execution: use provided seed/key if given, otherwise generate fresh ones
    run_seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
    key = int(args.key, 0) if args.key is not None else secrets.randbits(80)

    print("PRESENT-64 dataset generator")
    print("run_seed =", run_seed)
    print("key =", hex(key))

    # =========================
    # load plaintexts
    # =========================

    plaintexts = []

    try:
        with open(args.plaintext_file) as f:
            for line in f:
                plaintexts.append(int(line.strip(), 16))

        if len(plaintexts) < args.n_real:
            raise ValueError("not enough plaintexts")

        print("loaded plaintexts:", len(plaintexts))

    except FileNotFoundError:
        plaintexts = None
        print("plaintext file not found -> random plaintexts")

    # =========================
    # SINGLE
    # =========================

    if args.mode in ["single", "both"]:
        delta = int(args.delta, 16)

        X, y = generate_dataset_single(
            args.n_real,
            args.n_random,
            delta,
            key,
            rounds,
            run_seed,
            plaintexts,
        )

        fname = f"present64_R1_r{rounds}_s{run_seed}.npz"

        np.savez_compressed(
            fname,
            X=X,
            y=y,
            key=np.array([key], dtype=object),
            seed=np.array([run_seed], dtype=np.uint32),
            rounds=np.array([rounds], dtype=np.uint16),
            delta=np.array([delta], dtype=object),
        )

        print("saved", fname, "shape", X.shape)

    # =========================
    # MULTI
    # =========================

    if args.mode in ["multi", "both"]:
        deltas = [int(x, 16) for x in args.deltas.split(",")]

        X, y = generate_dataset_multi(
            args.n_real,
            args.n_random,
            deltas,
            key,
            rounds,
            run_seed,
            plaintexts,
        )

        fname = f"present64_R2_r{rounds}_s{run_seed}.npz"

        np.savez_compressed(
            fname,
            X=X,
            y=y,
            key=np.array([key], dtype=object),
            seed=np.array([run_seed], dtype=np.uint32),
            rounds=np.array([rounds], dtype=np.uint16),
            deltas=np.array(deltas, dtype=object),
        )

        print("saved", fname, "shape", X.shape)
