# gift64_cipher.py

import random

# --- GIFT-64 S-box ---
SBOX = [0x1, 0xA, 0x4, 0xC, 0x6, 0xF, 0x3, 0x9, 0x2, 0xD, 0xB, 0x7, 0x5, 0x0, 0x8, 0xE]

# --- P-box permutation ---
PBOX = [0] * 64
for i in range(64):
    PBOX[i] = (16 * (i % 4)) + (i // 4)

# --- Round constants for 28 rounds (GIFT-64) ---
ROUND_CONSTANTS = [
    0x01,
    0x03,
    0x07,
    0x0F,
    0x1F,
    0x3E,
    0x3D,
    0x3B,
    0x37,
    0x2F,
    0x1E,
    0x3C,
    0x39,
    0x33,
    0x27,
    0x0E,
    0x1D,
    0x3A,
    0x35,
    0x2B,
    0x16,
    0x2C,
    0x18,
    0x30,
    0x21,
    0x02,
    0x05,
    0x0B,
]


def permute(state: int) -> int:
    """Bit permutation layer of GIFT-64."""
    out = 0
    for i in range(64):
        bit = (state >> i) & 1
        out |= bit << PBOX[i]
    return out


def substitute(state: int) -> int:
    """Substitution layer of GIFT-64 (16 nibbles)."""
    out = 0
    for i in range(16):  # 16 nibbles
        nibble = (state >> (4 * i)) & 0xF
        out |= SBOX[nibble] << (4 * i)
    return out


def get_round_key(key: int) -> int:
    """Extract 32-bit round key from 128-bit master key."""
    u = (key >> 112) & 0xFFFF
    v = (key >> 80) & 0xFFFF
    return (u << 16) | v


def update_key(key: int, round_num: int) -> int:
    """GIFT-64 key schedule update."""
    # Split into 8 × 16-bit words K0..K7
    k = [(key >> (112 - 16 * i)) & 0xFFFF for i in range(8)]

    # Rotate K1 >>> 2, K0 >>> 12
    k[1] = ((k[1] >> 2) | (k[1] << 14)) & 0xFFFF
    k[0] = ((k[0] >> 12) | (k[0] << 4)) & 0xFFFF

    rc = ROUND_CONSTANTS[round_num]
    # RC bit 0 into MSB of K0
    k[0] ^= (rc & 0x01) << 15
    # RC bits 1..5 into bits 11..7 of K4
    for i in range(5):
        bit = (rc >> (i + 1)) & 1
        k[4] ^= bit << (11 - i)

    new_key = 0
    for i in range(8):
        new_key |= k[i] << (112 - 16 * i)
    return new_key


def gift64_encrypt(
    plaintext: int, master_key: int, r_start: int = 0, r_end: int = 10
) -> int:
    """
    Encrypt 64-bit plaintext under 128-bit key for rounds [r_start, r_end).
    """
    state = plaintext
    key = master_key

    # Pre-advance key to r_start
    for r in range(r_start):
        key = update_key(key, r)

    for r in range(r_start, r_end):
        rk = get_round_key(key)
        # AddRoundKey: rk bits into even/odd state bits
        for i in range(16):
            state ^= ((rk >> i) & 1) << (2 * i)
            state ^= ((rk >> (i + 16)) & 1) << (2 * i + 1)

        state = substitute(state)
        state = permute(state)
        key = update_key(key, r)

    return state


if __name__ == "__main__":
    random.seed(0)
    key = random.getrandbits(128)
    pt = random.getrandbits(64)
    ct = gift64_encrypt(pt, key, 0, 4)
    print("Sample ct (4 rounds):", hex(ct))
