"""CPU screen for ranked-queue rank 10 (curve-logit recalibration, shift + temperature, KD-off re-screen).

Uses the records screen_pass.py wrote for realcyc (p = curve prob at the label t, y, u = user). Fits
  logit p' = a * logit p + b
on HALF the users (by USER, both folds), minimising the BY-USER-MEAN BCE (the metric, not the row-pooled
loss), and scores the held-out half's by-user-mean BCE improvement. Reports the shift-only (a = 1) and
the shift+temperature fits, and the calibration gap by review index (a monotone gap argues for the
TRAINED route -- the head can condition the shift -- a flat one for post-hoc).
Kill rule (literature.md / steelman.md): held-out by-user prize < +0.0001 ahead OR |train-user gap| < 0.001.
Usage: recal_probe.py [records.npz]
"""
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/proposals_2026-09-04/screen_records.npz"
d = np.load(path)
p, y, u, n = d["p"], d["y"], d["u"], d["n"]
users = sorted(set(u.tolist()))
lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def by_user_bce(z, y, u, users):
    out = []
    for uid in users:
        m = u == uid
        q = 1 / (1 + np.exp(-z[m]))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        out.append(-(y[m] * np.log(q) + (1 - y[m]) * np.log(1 - q)).mean())
    return float(np.mean(out))


def fit(lp, y, u, users, temperature):
    # coarse-to-fine grid on (a, b); the objective is smooth and 1-2 dimensional
    best = (by_user_bce(lp, y, u, users), 1.0, 0.0)
    a_grid = np.linspace(0.7, 1.3, 25) if temperature else [1.0]
    for a in a_grid:
        for b in np.linspace(-0.6, 0.6, 49):
            v = by_user_bce(a * lp + b, y, u, users)
            if v < best[0]:
                best = (v, a, b)
    # refine
    v0, a0, b0 = best
    for a in (np.linspace(a0 - 0.03, a0 + 0.03, 13) if temperature else [a0]):
        for b in np.linspace(b0 - 0.03, b0 + 0.03, 13):
            v = by_user_bce(a * lp + b, y, u, users)
            if v < best[0]:
                best = (v, a, b)
    return best


print(f"{len(users)} users, {len(p):,} rows; train-user calibration gap mean(y)-mean(p) = {np.mean([y[u==k].mean()-p[u==k].mean() for k in users]):+.5f}")
rng = np.random.RandomState(0)
order = list(users); rng.shuffle(order)
folds = [order[: len(order) // 2], order[len(order) // 2:]]
for temperature in (False, True):
    gains = []
    for k in (0, 1):
        tr, te = folds[k], folds[1 - k]
        mtr = np.isin(u, tr)
        _, a, b = fit(lp[mtr], y[mtr], u[mtr], tr, temperature)
        mte = np.isin(u, te)
        base = by_user_bce(lp[mte], y[mte], u[mte], te)
        new = by_user_bce(a * lp[mte] + b, y[mte], u[mte], te)
        gains.append(base - new)
        print(f"  {'shift+temp' if temperature else 'shift only '} fold {k}: fit a={a:.3f} b={b:+.3f} on {len(tr)} users -> held-out by-user BCE {base:.6f} -> {new:.6f}  prize {base - new:+.6f}")
    print(f"  {'shift+temp' if temperature else 'shift only '}: MEAN held-out prize {np.mean(gains):+.6f}   (kill line +0.0001)")
print("calibration gap by review index (mean y - mean p):")
for lo, hi, lab in ((0, 1, "1st pred"), (1, 2, "2nd"), (2, 4, "3rd-4th"), (4, 8, "5th-8th"), (8, 16, "9th-16th"), (16, 10**9, "17th+")):
    m = (n >= lo) & (n < hi)
    print(f"  {lab:<9} n={int(m.sum()):>8,}  gap {y[m].mean() - p[m].mean():+.5f}")
