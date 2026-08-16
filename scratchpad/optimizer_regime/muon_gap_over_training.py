#!/usr/bin/env python
"""Does Muon's advantage over AdamW shrink over a run? Reads step traces; seconds, no GPU.

Andrew 2026-08-16: "the first time we tried Muon it was way better than Adam initially, but only
mildly better at the end. Muon seems to be optimized for speedrunning: getting to a fixed loss in
as few epochs as possible, whereas we do the opposite."

That is a testable claim and the archive can settle it, because iter 29 (Muon) and iter 26 (AdamW)
are a MATCHED PAIR -- diffing their runners shows the only difference is the three RWKV_MUON_* env
vars, same arch, PAVA lambda, GRU head, data, LR, wd, clip, seed and budget.

⚠ THE OBVIOUS CONTROL IS THE WRONG ONE. `champ5k_plain` looks like the AdamW counterpart and is
not: it lacks RWKV_PAVA_LAMBDA and RWKV_GRU_HEAD, so it differs in the ahead objective AND the
head. Using it inflates the early imm gap and produces a confident wrong story. Diff the runners
before trusting any pair -- this is the same trap as iter 47's wrong-checkpoint control.

Usage: muon_gap_over_training.py [muon_trace] [adamw_trace]
ASCII output only.
"""
import io
import json
import sys

import numpy as np

MU = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/iter29_muon/iter29_muon_ws_trace.jsonl"
AD = sys.argv[2] if len(sys.argv) > 2 else "scratchpad/iter26_gru3/iter26_gru3_ws_trace.jsonl"


def load(p):
    d = {}
    for line in io.open(p, encoding="utf-8"):
        r = json.loads(line)
        d[r["step"]] = (r["ahead"], r["imm"])
    return d


M, A = load(MU), load(AD)
c = sorted(set(M) & set(A))
m = np.array([M[s] for s in c])
a = np.array([A[s] for s in c])
st = np.array(c)
d = a - m  # positive = Muon better on TRAIN loss

print(f"muon={MU}\nadamw={AD}\npaired steps: {len(c)} ({st[0]}..{st[-1]})\n")
print(f"  {'steps':>17s} {'ahead gap':>11s} {'imm gap':>11s}   (+ = Muon better on TRAIN loss)")
n = len(c)
for i in range(10):
    lo, hi = i * n // 10, (i + 1) * n // 10
    print(f"  {st[lo]:>7d}-{st[hi-1]:<9d} {d[lo:hi, 0].mean():11.5f} {d[lo:hi, 1].mean():11.5f}")
f, l = d[: n // 10], d[-n // 10:]
print(f"\n  first decile: ahead {f[:, 0].mean():+.5f}   imm {f[:, 1].mean():+.5f}")
print(f"  last  decile: ahead {l[:, 0].mean():+.5f}   imm {l[:, 1].mean():+.5f}")
print("\n  Compare against the EVAL endpoints: if train converges while eval stays apart, the")
print("  optimizer is acting as a REGULARIZER, not as a faster descent.")
