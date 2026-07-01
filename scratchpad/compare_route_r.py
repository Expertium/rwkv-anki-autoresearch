"""Route-R comparison: 8192-chunk (sc8k) vs 65536-chunk (base65k) COLD WS training, users 101-200.
Reports by-user-mean LogLoss (ahead + imm) for both + the size-identity check (must match -- same test
set). Speed is read separately from the training logs (loss_n/sec + train_elapsed_min). The question:
does the smaller cold chunk (B~4) cost accuracy vs the big cold chunk (B~1)? If the gap is within the
~0.0018 augmentation noise, smaller cold chunks are ~free and the intricate stateful carry is low-ROI."""
import json


def load(f):
    d = {}
    for line in open(f):
        r = json.loads(line)
        d[r["user"]] = (r["size"], r["metrics"]["LogLoss"])
    return d


MODELS = [
    ("base65k (65536-chunk, B~1)", "RWKV-r-base65k", "RWKV-P-r-base65k"),
    ("sc8k    ( 8192-chunk, B~4)", "RWKV-r-sc8k", "RWKV-P-r-sc8k"),
]
data = {}
for name, fa, fi in MODELS:
    try:
        data[name] = (load(f"result/{fa}.jsonl"), load(f"result/{fi}.jsonl"))
    except FileNotFoundError as e:
        print("MISSING:", name, "->", e)

if len(data) == 2:
    users = sorted(next(iter(data.values()))[0])
    sizes_ok = all(
        len({d[0][u][0] for d in data.values()}) == 1 and len({d[1][u][0] for d in data.values()}) == 1
        for u in users if all(u in d[0] and u in d[1] for d in data.values())
    )
    print(f"users evaluated: {len(users)}   SIZE IDENTICAL across both: {sizes_ok}")
    print(f"\n=== by-user mean LogLoss on 101-200 ===\n{'model':30} {'ahead':>10} {'imm':>10}")
    res = {}
    for name in [m[0] for m in MODELS]:
        if name in data:
            d = data[name]
            ma = sum(v[1] for v in d[0].values()) / len(d[0])
            mi = sum(v[1] for v in d[1].values()) / len(d[1])
            res[name] = (ma, mi)
            print(f"{name:30} {ma:10.6f} {mi:10.6f}")
    keys = list(res)
    if len(keys) == 2:
        da = res[keys[1]][0] - res[keys[0]][0]
        di = res[keys[1]][1] - res[keys[0]][1]
        print(f"\nDELTA (sc8k - base65k):   ahead {da:+.6f}   imm {di:+.6f}")
        print("(augmentation run-to-run noise ~0.0018 imm / 0.0006 ahead -- a gap within that is ~free)")
