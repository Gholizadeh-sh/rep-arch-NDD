# Present64.py
import random

# =========================
# PRESENT S-box
# =========================

SBOX = [
    0xC,
    0x5,
    0x6,
    0xB,
    0x9,
    0x0,
    0xA,
    0xD,
    0x3,
    0xE,
    0xF,
    0x8,
    0x4,
    0x7,
    0x1,
    0x2,
]

# permutation layer
PBOX = [0] * 64
for i in range(63):
    PBOX[i] = (16 * i) % 63
PBOX[63] = 63


# =========================
# S-box layer
# =========================


def sbox_layer(state: int) -> int:

    out = 0

    for i in range(16):

        nibble = (state >> (4 * i)) & 0xF
        s = SBOX[nibble]

        out |= s << (4 * i)

    return out


# =========================
# permutation layer
# =========================


def p_layer(state: int) -> int:

    out = 0

    for i in range(64):

        bit = (state >> i) & 1
        out |= bit << PBOX[i]

    return out


# =========================
# key schedule (80-bit)
# =========================


def generate_round_keys_80(master_key: int, rounds: int):

    key = master_key & ((1 << 80) - 1)

    round_keys = []

    for r in range(1, rounds + 1):

        # round key = leftmost 64 bits
        round_key = key >> 16
        round_keys.append(round_key & 0xFFFFFFFFFFFFFFFF)

        # rotate key left by 61 bits
        key = ((key << 61) & ((1 << 80) - 1)) | (key >> 19)

        # apply S-box to MS nibble
        ms_nibble = (key >> 76) & 0xF
        key &= ~(0xF << 76)
        key |= SBOX[ms_nibble] << 76

        # XOR round counter
        key ^= r << 15

    # final whitening key
    round_keys.append((key >> 16) & 0xFFFFFFFFFFFFFFFF)

    return round_keys


# =========================
# PRESENT-64 encryption
# =========================


def encrypt_present64_80(plaintext: int, master_key: int, rounds: int = 31):

    state = plaintext & 0xFFFFFFFFFFFFFFFF

    round_keys = generate_round_keys_80(master_key, rounds)

    for r in range(rounds):

        # AddRoundKey
        state ^= round_keys[r]

        # S-box layer
        state = sbox_layer(state)

        # permutation layer
        state = p_layer(state)

    # final AddRoundKey
    state ^= round_keys[rounds]

    return state & 0xFFFFFFFFFFFFFFFF


# =========================
# test
# =========================

if __name__ == "__main__":

    mk = random.getrandbits(80)
    pt = random.getrandbits(64)

    ct = encrypt_present64_80(pt, mk, rounds=10)

    print(f"PT = {pt:016X}")
    print(f"CT = {ct:016X}")
