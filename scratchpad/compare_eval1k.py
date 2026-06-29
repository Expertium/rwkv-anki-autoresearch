"""Compare OLD (RWKV_trained_on_5000_10000, 2.76M) vs NEW (champion iter36, 192.8k) on users
1001-2000: verify per-user `size` (equalized review count) is IDENTICAL between the two models'
evals (proof the preprocessing is the same), then report by-user-mean LogLoss for both modes
(ahead = forgetting-curve, imm = immediate) + RMSE(bins). Writes a per-user CSV."""
import csv
import json


def load(f):
    d = {}
    for line in open(f):
        r = json.loads(line)
        m = r["metrics"]
        d[r["user"]] = (r["size"], m["LogLoss"], m["RMSE(bins)"])
    return d


oa = load("result/RWKV-old-1k.jsonl")   # old, ahead
na = load("result/RWKV-new-1k.jsonl")   # new, ahead
oi = load("result/RWKV-P-old-1k.jsonl")  # old, imm
ni = load("result/RWKV-P-new-1k.jsonl")  # new, imm
users = sorted(oa)

# --- size identity (the key preprocessing-sameness check) ---
mism = [u for u in users if not (oa[u][0] == na[u][0] == oi[u][0] == ni[u][0])]
print(f"users evaluated: {len(users)}  (old {len(oa)}, new {len(na)})")
print(f"SIZE IDENTICAL across old/new and both modes: {len(mism) == 0}  (mismatches: {len(mism)})")
if mism:
    for u in mism[:10]:
        print(f"  user {u}: sizes old_ahead={oa[u][0]} new_ahead={na[u][0]} old_imm={oi[u][0]} new_imm={ni[u][0]}")


def mean(d, i):
    return sum(v[i] for v in d.values()) / len(d)


print("\n=== by-user mean LogLoss (each user weighted equally) ===")
print(f"  OLD (2.76M, trained 5000-10000):  ahead {mean(oa,1):.6f}   imm {mean(oi,1):.6f}")
print(f"  NEW (192.8k champion, trained 1-100): ahead {mean(na,1):.6f}   imm {mean(ni,1):.6f}")
print(f"  delta (new-old):                  ahead {mean(na,1)-mean(oa,1):+.6f}   imm {mean(ni,1)-mean(oi,1):+.6f}")
print("\n=== by-user mean RMSE(bins) ===")
print(f"  OLD:  ahead {mean(oa,2):.5f}   imm {mean(oi,2):.5f}")
print(f"  NEW:  ahead {mean(na,2):.5f}   imm {mean(ni,2):.5f}")

with open("result/compare_eval1k.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["user", "size", "ahead_old", "ahead_new", "imm_old", "imm_new",
                "rmsebins_ahead_old", "rmsebins_ahead_new", "rmsebins_imm_old", "rmsebins_imm_new"])
    for u in users:
        w.writerow([u, oa[u][0], oa[u][1], na[u][1], oi[u][1], ni[u][1],
                    oa[u][2], na[u][2], oi[u][2], ni[u][2]])
print("\nper-user CSV -> result/compare_eval1k.csv")
