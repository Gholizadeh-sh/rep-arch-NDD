import random

N = 2**15

output_file = "plaintexts64.txt"

seed = 42
random.seed(seed)

plaintexts = [random.getrandbits(64) for _ in range(N)]

with open(output_file, "w") as f:
    for p in plaintexts:
        f.write(f"{p:016x}\n")

print(f"{N} plaintexts saved to {output_file}")
