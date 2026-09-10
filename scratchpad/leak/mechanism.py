"""Does the in-session gap (show_k - answer_{k-1}) carry the CURRENT review's outcome via the duration cap?
Raw -id parquet, read only. show = id - taken (as the builder does); answer_{k-1} = show_{k-1} + dur_{k-1}."""
import sys
import numpy as np
import pandas as pd

ROOT = "C:/Users/Andrew/anki-revlogs-10k-id/revlogs"
users = [int(u) for u in sys.argv[1:]] or list(range(5001, 5041))
rows = []
for u in users:
    try:
        df = pd.read_parquet(f"{ROOT}/user_id={u}/data.parquet", columns=["review_time", "duration", "rating"])
    except Exception as e:
        continue
    df = df.sort_values("review_time", kind="stable")
    rt = df["review_time"].to_numpy(np.int64)
    du = df["duration"].to_numpy(np.int64)
    rating = df["rating"].to_numpy()
    if len(rt) < 100:
        continue
    gap = np.full(len(rt), np.nan)
    gap[1:] = (rt[1:] - (rt[:-1] + du[:-1])) / 1000.0
    cap = int(du.max())
    rows.append(pd.DataFrame({"user": u, "gap": gap, "dur": du, "cap": cap,
                              "capped": du == cap, "fail": rating == 1}))
d = pd.concat(rows, ignore_index=True)
d = d[d["gap"].notna()]
print(f"users {d.user.nunique()}  reviews {len(d):,}")
print("per-user cap (max duration) value counts:", d.groupby("user")["cap"].first().value_counts().head(5).to_dict())
print(f"P(capped) {d.capped.mean():.4f}   P(fail|capped) {d[d.capped].fail.mean():.4f}   "
      f"P(fail|not capped) {d[~d.capped].fail.mean():.4f}")
bins = [-1e9, 0, 1, 2, 3, 5, 10, 30, 60, 300, 1800, 3 * 3600, 1e12]
lab = ["<=0", "0-1s", "1-2s", "2-3s", "3-5s", "5-10s", "10-30s", "30-60s", "1-5m", "5-30m", "30m-3h", ">3h"]
d["b"] = pd.cut(d["gap"], bins=bins, labels=lab)
t = d.groupby("b", observed=False).agg(n=("fail", "size"), share=("fail", "size"), p_fail=("fail", "mean"),
                                        p_capped=("capped", "mean"))
t["share"] = t["n"] / t["n"].sum()
print(t.to_string(float_format=lambda x: f"{x:.4f}"))
ins = d[(d.gap > 0) & (d.gap < 1800)]
print("\nwithin 30 min, split by capped:")
print(ins.groupby("capped").agg(n=("fail", "size"), median_gap=("gap", "median"), p90_gap=("gap", lambda s: s.quantile(.9)),
                                p_fail=("fail", "mean")).to_string(float_format=lambda x: f"{x:.3f}"))

print("\nP(fail) by gap bucket, UNCAPPED reviews only (a capped review cannot explain these):")
u = d[~d.capped]
t2 = u.groupby("b", observed=False).agg(n=("fail", "size"), p_fail=("fail", "mean"))
print(t2.to_string(float_format=lambda x: f"{x:.4f}"))

def H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))
base = H(d.fail.mean())
g = d.groupby("b", observed=False)["fail"].agg(["mean", "size"])
cond = (g["size"] * H(g["mean"])).sum() / g["size"].sum()
print(f"\nlogloss of the base rate {base:.5f}; knowing only the gap bucket {cond:.5f}; gain {base - cond:.5f} nats")
lo = d[(d.gap > 1) & (d.gap < 1800)]
print(f"rows with a 1 s - 30 min gap: {len(lo) / len(d):.4f} of all reviews")
