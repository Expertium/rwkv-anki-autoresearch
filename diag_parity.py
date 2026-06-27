"""Diagnose Rust-vs-Python divergence: is it an early-review bug or late-review drift?"""
import json
from pathlib import Path

import numpy as np

REF = Path("reference")
u = 107
meta = json.load(open(REF / f"trace_user_{u}.json"))
rust = json.load(open(REF / f"rust_pred_{u}.json"))

py_imm = {int(k): float(v) for k, v in meta["py_pred_imm"].items()}
py_ahead = {int(k): float(v) for k, v in meta["py_pred_ahead"].items()}
rth = rust["review_th"]
r_imm = {rt: p for rt, p in zip(rth, rust["pred_imm"])}
r_ahead = {rt: p for rt, p in zip(rth, rust["pred_ahead"]) if p is not None}

# imm exists for every review; sort by review_th (== order)
order = sorted(py_imm.keys())
diffs = np.array([abs(r_imm[rt] - py_imm[rt]) for rt in order])
print(f"user {u}: {len(order)} reviews")
print("imm diff: first 8 reviews:",
      [f"{abs(r_imm[rt]-py_imm[rt]):.2e}" for rt in order[:8]])
print(f"imm diff: mean {diffs.mean():.2e}  median {np.median(diffs):.2e}  "
      f"p99 {np.percentile(diffs,99):.2e}  max {diffs.max():.2e}")

# correlation of diff with position (drift signature)
pos = np.arange(len(order))
half = len(order) // 2
print(f"imm diff: first-half mean {diffs[:half].mean():.2e}  "
      f"second-half mean {diffs[half:].mean():.2e}")

# worst offenders
worst = np.argsort(diffs)[-6:][::-1]
print("imm worst reviews (idx, review_th, py, rust, diff):")
for w in worst:
    rt = order[w]
    print(f"  idx {w:5d}  rt {rt:6d}  py {py_imm[rt]:.5f}  rust {r_imm[rt]:.5f}  d {diffs[w]:.3e}")

# ahead
ao = sorted(py_ahead.keys())
ad = np.array([abs(r_ahead[rt] - py_ahead[rt]) for rt in ao])
print(f"\nahead diff: mean {ad.mean():.2e}  median {np.median(ad):.2e}  "
      f"p99 {np.percentile(ad,99):.2e}  max {ad.max():.2e}")
