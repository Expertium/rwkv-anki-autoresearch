"""Champion aug-off re-measure: two identical runs (1234 fixed aug seed, determinism on).
Reports by-user-mean LogLoss (ahead + imm) for each run, the run-to-run variance |run1-run2|,
and the size-identity check. With augmentation disabled the variance should be ~0 (ideally bit-equal),
validating that the 0.0003 acceptance gate is usable. The run1 numbers are the gate REFERENCE
(fp32 champion, aug-off); the deployed (quant+low-rank) number is measured separately via Rust."""
import json


def load(f):
    d = {}
    for line in open(f):
        r = json.loads(line)
        d[r["user"]] = (r["size"], r["metrics"]["LogLoss"])
    return d


RUNS = [
    ("run1", "RWKV-champoff1", "RWKV-P-champoff1"),
    ("run2", "RWKV-champoff2", "RWKV-P-champoff2"),
]
data = {}
for name, fa, fi in RUNS:
    try:
        data[name] = (load(f"result/{fa}.jsonl"), load(f"result/{fi}.jsonl"))
    except FileNotFoundError as e:
        print("MISSING:", name, "->", e)

if len(data) == 2:
    users = sorted(set(data["run1"][0]) & set(data["run2"][0]))
    sizes_ok = all(
        data["run1"][0][u][0] == data["run2"][0][u][0] and data["run1"][1][u][0] == data["run2"][1][u][0]
        for u in users
    )
    print(f"users evaluated: {len(users)}   SIZE IDENTICAL run1 vs run2: {sizes_ok}")
    print(f"\n=== by-user mean LogLoss on 101-200 ===\n{'run':10} {'ahead':>10} {'imm':>10}")
    res = {}
    for name, _, _ in RUNS:
        if name in data:
            d = data[name]
            ma = sum(v[1] for v in d[0].values()) / len(d[0])
            mi = sum(v[1] for v in d[1].values()) / len(d[1])
            res[name] = (ma, mi)
            print(f"{name:10} {ma:10.6f} {mi:10.6f}")
    if len(res) == 2:
        da = abs(res["run1"][0] - res["run2"][0])
        di = abs(res["run1"][1] - res["run2"][1])
        # per-user max abs diff (a tighter variance probe than the mean)
        pa = max(abs(data["run1"][0][u][1] - data["run2"][0][u][1]) for u in users)
        pi = max(abs(data["run1"][1][u][1] - data["run2"][1][u][1]) for u in users)
        print(f"\nVARIANCE |run1-run2| (mean):   ahead {da:.6f}   imm {di:.6f}")
        print(f"VARIANCE per-user MAX abs:     ahead {pa:.6f}   imm {pi:.6f}")
        print("\nGATE REFERENCE (fp32 champion, aug-off) = run1:")
        print(f"   ahead {res['run1'][0]:.6f}   imm {res['run1'][1]:.6f}")
        print("(acceptance gate = 0.0003; variance must be << that for a single-run win to be real)")
