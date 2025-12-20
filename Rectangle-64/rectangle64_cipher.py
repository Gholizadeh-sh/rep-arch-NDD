# rectangle64.py
import random

# =========================
#  S-box و شیفت‌رو
# =========================

# S-box از مقاله NIST RECTANGLE (جدول S(x))
# x : 0..F  ->  S(x)
SBOX = [
    0x6,
    0x5,
    0xC,
    0xA,
    0x1,
    0xE,
    0x7,
    0x9,
    0xB,
    0x0,
    0x3,
    0xD,
    0x8,
    0xF,
    0x4,
    0x2,
]


def rol16(x: int, r: int) -> int:
    """چرخش چپ روی ۱۶ بیت."""
    r %= 16
    return ((x << r) & 0xFFFF) | (x >> (16 - r))


# =========================
#  توابع روی حالت ۶۴ بیتی
# =========================


def split_state_rows(state: int):
    """
    وضعیت ۶۴بیتی را به ۴ کلمه‌ی ۱۶بیتی (سطرها) تبدیل می‌کند.
    row0 = بیت‌های ۰..۱۵ (LSB)
    row1 = بیت‌های ۱۶..۳۱
    row2 = بیت‌های ۳۲..۴۷
    row3 = بیت‌های ۴۸..۶۳
    """
    row0 = state & 0xFFFF
    row1 = (state >> 16) & 0xFFFF
    row2 = (state >> 32) & 0xFFFF
    row3 = (state >> 48) & 0xFFFF
    return row0, row1, row2, row3


def combine_state_rows(row0: int, row1: int, row2: int, row3: int) -> int:
    """۴ سطر ۱۶بیتی را به یک وضعیت ۶۴بیتی تبدیل می‌کند."""
    return (row3 << 48) | (row2 << 32) | (row1 << 16) | row0


def subcolumn(state: int) -> int:
    """
    مرحله SubColumn:
    روی هر ستون (۴ بیت عمودی) یک S-box اعمال می‌کند.
    ورودی S-box: Col(j) = a3,j || a2,j || a1,j || a0,j
    خروجی: b3,j .. b0,j و جایگزینی در همان ستون.
    """
    row0, row1, row2, row3 = split_state_rows(state)

    out0 = out1 = out2 = out3 = 0

    for j in range(16):
        a0 = (row0 >> j) & 1
        a1 = (row1 >> j) & 1
        a2 = (row2 >> j) & 1
        a3 = (row3 >> j) & 1

        x = (a3 << 3) | (a2 << 2) | (a1 << 1) | a0
        y = SBOX[x]

        b0 = y & 1
        b1 = (y >> 1) & 1
        b2 = (y >> 2) & 1
        b3 = (y >> 3) & 1

        out0 |= b0 << j
        out1 |= b1 << j
        out2 |= b2 << j
        out3 |= b3 << j

    return combine_state_rows(out0, out1, out2, out3)


def shiftrow(state: int) -> int:
    """
    مرحله ShiftRow:
    - سطر ۰: بدون چرخش
    - سطر ۱: چرخش چپ ۱ بیت
    - سطر ۲: چرخش چپ ۱۲ بیت
    - سطر ۳: چرخش چپ ۱۳ بیت
    """
    row0, row1, row2, row3 = split_state_rows(state)
    row0p = row0
    row1p = rol16(row1, 1)
    row2p = rol16(row2, 12)
    row3p = rol16(row3, 13)
    return combine_state_rows(row0p, row1p, row2p, row3p)


# =========================
#  Key schedule برای ۸۰بیت
# =========================


def generate_round_constants(num_rounds: int):
    """
    تولید RC[i] با LFSR ۵ بیتی مطابق مقاله:
    - مقدار اولیه RC[0] = 0x1
    - در هر راند، (rc4..rc0) یک بیت به چپ شیفت می‌شود
      و rc0 جدید = rc4 XOR rc2
    """
    rc = 0x1
    rcs = []
    for _ in range(num_rounds):
        rcs.append(rc & 0x1F)
        rc4 = (rc >> 4) & 1
        rc2 = (rc >> 2) & 1
        rc = ((rc << 1) & 0x1E) | (rc4 ^ rc2)
    return rcs


def init_key_rows_80(master_key: int):
    """
    کلید ۸۰بیتی را به ۵ سطر ۱۶بیتی تبدیل می‌کند.
    فرض: master_key به صورت v79..v0 است و v0 در LSB قرار دارد.
    Row0 = v15..v0 = بیت‌های ۰..۱۵
    Row1 = v31..v16
    ...
    Row4 = v79..v64
    """
    rows = []
    for i in range(5):
        rows.append((master_key >> (16 * i)) & 0xFFFF)
    # rows[0] = Row0, ..., rows[4] = Row4
    return rows  # لیست ۵ تایی


def form_round_key_from_rows(rows):
    """
    طبق مشخصات:
    Ki = Row3 || Row2 || Row1 || Row0 (۶۴ بیت)
    ما Row0 را در پایین (LSB) و Row3 را در بالا قرار می‌دهیم.
    """
    Row0, Row1, Row2, Row3, _ = rows
    return combine_state_rows(Row0, Row1, Row2, Row3)


def update_key_rows_80(rows, rc: int):
    """
    آپدیت رجیستر کلید ۸۰بیتی در یک راند طبق مقاله:
      1) S-box روی ۴ ستون راست بالا
      2) Feistel روی Row0..Row4
      3) XOR کردن RC[i] با ۵ بیت پایین Row0
    """
    Row0, Row1, Row2, Row3, Row4 = rows

    # 1) S-box روی ۴ ستون راست (j = 0..3)
    for j in range(4):
        k0 = (Row0 >> j) & 1
        k1 = (Row1 >> j) & 1
        k2 = (Row2 >> j) & 1
        k3 = (Row3 >> j) & 1

        x = (k3 << 3) | (k2 << 2) | (k1 << 1) | k0
        y = SBOX[x]

        b0 = y & 1
        b1 = (y >> 1) & 1
        b2 = (y >> 2) & 1
        b3 = (y >> 3) & 1

        # جایگزینی در همان ستون j
        if b0 != k0:
            Row0 ^= 1 << j
        if b1 != k1:
            Row1 ^= 1 << j
        if b2 != k2:
            Row2 ^= 1 << j
        if b3 != k3:
            Row3 ^= 1 << j

    # 2) Feistel
    new_Row0 = rol16(Row0, 8) ^ Row1
    new_Row1 = Row2
    new_Row2 = Row3
    new_Row3 = rol16(Row3, 12) ^ Row4
    new_Row4 = Row0

    # 3) XOR با RC روی بیت‌های ۰..۴ Row0
    new_Row0 ^= rc & 0x1F

    return [
        new_Row0 & 0xFFFF,
        new_Row1 & 0xFFFF,
        new_Row2 & 0xFFFF,
        new_Row3 & 0xFFFF,
        new_Row4 & 0xFFFF,
    ]


def generate_round_keys_80(master_key: int, rounds: int):
    """
    تولید کلیدهای راند برای نسخه‌ی ۸۰بیتی و r راند:
    - K[0]..K[r-1] برای AddRoundKey در هر راند
    - K[r] برای AddRoundKey نهایی
    """
    rows = init_key_rows_80(master_key)
    rcs = generate_round_constants(rounds)

    round_keys = []

    for i in range(rounds):
        Ki = form_round_key_from_rows(rows)
        round_keys.append(Ki)
        rows = update_key_rows_80(rows, rcs[i])

    # کلید نهایی بعد از آپدیت آخر
    K_final = form_round_key_from_rows(rows)
    round_keys.append(K_final)

    return round_keys  # طول: rounds+1


# =========================
#  رمزگذاری
# =========================


def encrypt_rect64_80(plaintext: int, master_key: int, rounds: int = 6) -> int:
    """
    رمزگذاری RECTANGLE-80 برای r راند:
    برای هر راند: AddRoundKey, SubColumn, ShiftRow
    در پایان: AddRoundKey نهایی.
    """
    state = plaintext & 0xFFFFFFFFFFFFFFFF
    round_keys = generate_round_keys_80(master_key, rounds)

    for r in range(rounds):
        # AddRoundKey
        state ^= round_keys[r]
        # SubColumn
        state = subcolumn(state)
        # ShiftRow
        state = shiftrow(state)

    # AddRoundKey نهایی
    state ^= round_keys[rounds]
    return state & 0xFFFFFFFFFFFFFFFF


if __name__ == "__main__":
    # یک تست خیلی ساده (فقط برای چک کردن سینتکس و اجرا)
    mk = random.getrandbits(80)
    pt = random.getrandbits(64)
    ct = encrypt_rect64_80(pt, mk, rounds=4)
    print(f"PT = {pt:016X}")
    print(f"CT = {ct:016X}")
