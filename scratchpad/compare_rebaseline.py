"""Compare on users 1001-2000: OLD (2.76M, trained 5000-10000) vs NEW 5%-data champion (iter36, the
data-drop bug) vs NEW 100%-coverage re-baseline (run1, MAX=66000). Verifies per-user `size` is identical
across all (preprocessing-sameness), reports by-user-mean LogLoss (ahead + imm). Also reports the
run-to-run variance (run1 vs run2 on 101-200)."""
import json


def load(f):
    d = {}
    for line in open(f):
        r = json.loads(line)
        d[r["user"]] = (r["size"], r["metrics"]["LogLoss"])
    return d


MODELS = [
    ("OLD 2.76M fp32 (train 5000-10000)", "RWKV-old-1k", "RWKV-P-old-1k"),
    ("NEW champion iter45 fp32 (5%-data)", "RWKV-iter45-1k", "RWKV-P-iter45-1k"),
    ("NEW 100%-cov re-baseline fp32 (WS+decay)", "RWKV-rb-1k", "RWKV-P-rb-1k"),
]
data = {}
for name, fa, fi in MODELS:
    try:
        data[name] = (load(f"result/{fa}.jsonl"), load(f"result/{fi}.jsonl"))
    except FileNotFoundError as e:
        print("MISSING:", name, "->", e)

if data:
    users = sorted(next(iter(data.values()))[0])
    sizes_ok = all(
        len({d[0][u][0] for d in data.values()}) == 1 and len({d[1][u][0] for d in data.values()}) == 1
        for u in users
    )
    print(f"users evaluated: {len(users)}   SIZE IDENTICAL across all models+modes: {sizes_ok}")
    print(f"\n=== by-user mean LogLoss on 1001-2000 ===\n{'model':34} {'ahead':>10} {'imm':>10}")
    for name in [m[0] for m in MODELS]:
        if name in data:
            d = data[name]
            ma = sum(v[1] for v in d[0].values()) / len(d[0])
            mi = sum(v[1] for v in d[1].values()) / len(d[1])
            print(f"{name:34} {ma:10.6f} {mi:10.6f}")

# run-to-run variance on 101-200
try:
    def mean(f):
        r = [json.loads(l) for l in open(f)]
        return sum(x["metrics"]["LogLoss"] for x in r) / len(r)
    a1, a2 = mean("result/RWKV-rb1-100.jsonl"), mean("result/RWKV-rb2-100.jsonl")
    i1, i2 = mean("result/RWKV-P-rb1-100.jsonl"), mean("result/RWKV-P-rb2-100.jsonl")
    print(f"\n=== RUN-TO-RUN VARIANCE (run1 vs run2, 100 users 101-200; determinism ON, augmentation stochastic) ===")
    print(f"  ahead: run1 {a1:.6f}  run2 {a2:.6f}  |diff| {abs(a1-a2):.6f}")
    print(f"  imm:   run1 {i1:.6f}  run2 {i2:.6f}  |diff| {abs(i1-i2):.6f}")
except FileNotFoundError as e:
    print("\nvariance MISSING:", e)
