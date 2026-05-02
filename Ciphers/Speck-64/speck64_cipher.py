# speck64_128.py
import random

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
MASK128 = (1 << 128) - 1


def rol32(x: int, r: int) -> int:
    r %= 32
    return ((x << r) & MASK32) | (x >> (32 - r))


def ror32(x: int, r: int) -> int:
    r %= 32
    return (x >> r) | ((x << (32 - r)) & MASK32)


def split_block_lr(block: int):
    """
    Split 64-bit block into two 32-bit words:
    left = MSW, right = LSW
    """
    left = (block >> 32) & MASK32
    right = block & MASK32
    return left, right


def combine_block_lr(left: int, right: int) -> int:
    """
    Combine two 32-bit words into one 64-bit block.
    """
    return ((left & MASK32) << 32) | (right & MASK32)


def split_key_128(master_key: int):
    """
    Split 128-bit key into four 32-bit words:
    K = (K3, K2, K1, K0) in the paper,
    while this function returns [K0, K1, K2, K3].
    """
    master_key &= MASK128
    return [(master_key >> (32 * i)) & MASK32 for i in range(4)]


def speck_round(x: int, y: int, k: int):
    """
    One Speck64 round on 32-bit words:
        x = (ROR(x,8) + y) xor k
        y = ROL(y,3) xor x
    """
    x = ror32(x, 8)
    x = (x + y) & MASK32
    x ^= k
    y = rol32(y, 3)
    y ^= x
    return x, y


def generate_round_keys_128(master_key: int, rounds: int = 27):
    """
    Generate round keys for Speck64/128.
    Reference ordering matches the NSA implementation guide.
    """
    K0, K1, K2, K3 = split_key_128(master_key)

    A = K0
    B = K1
    C = K2
    D = K3

    round_keys = []
    i = 0

    while i < rounds:
        round_keys.append(A)
        if len(round_keys) == rounds:
            break
        B, A = speck_round(B, A, i)
        i += 1

        round_keys.append(A)
        if len(round_keys) == rounds:
            break
        C, A = speck_round(C, A, i)
        i += 1

        round_keys.append(A)
        if len(round_keys) == rounds:
            break
        D, A = speck_round(D, A, i)
        i += 1

    return round_keys


def encrypt_speck64_128(plaintext: int, master_key: int, rounds: int = 27) -> int:
    """
    Encrypt 64-bit plaintext under 128-bit key.
    Supports reduced-round experiments via `rounds`.
    """
    x, y = split_block_lr(plaintext & MASK64)
    round_keys = generate_round_keys_128(master_key, rounds)

    for rk in round_keys:
        x, y = speck_round(x, y, rk)

    return combine_block_lr(x, y) & MASK64


if __name__ == "__main__":
    # Standard Speck64/128 test vector from NSA guide
    pt = 0x3B7265747475432D
    key = 0x1B1A1918131211100B0A090803020100
    ct = encrypt_speck64_128(pt, key, rounds=27)

    print(f"PT = {pt:016X}")
    print(f"CT = {ct:016X}")
    # Expected: 8C6FA548454E028B

    # Random quick test
    mk = random.getrandbits(128)
    p = random.getrandbits(64)
    c = encrypt_speck64_128(p, mk, rounds=10)
    print(f"Random PT = {p:016X}")
    print(f"Random CT (10 rounds) = {c:016X}")
