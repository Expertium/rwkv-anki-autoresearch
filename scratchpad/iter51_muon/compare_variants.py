#!/usr/bin/env python
"""Three Newton-Schulz variants, scored on the directions that carry the momentum. CPU, ~1 min.

Established first (ns_error.py, where_is_the_error.py):
  * the 0.19-0.31 RMS claim reproduces (0.274 over ALL singular values);
  * it is NOT precision -- bf16 and fp32 agree to 0.01;
  * about half of it is the near-null tail, which no odd polynomial can lift in 5 steps and which
    we would not WANT lifted (it is noise amplification);
  * but a real 0.16 / 0.12 remains on the top 90% / 99% of energy, which IS fixable;
  * the energy-bearing interval after Frobenius normalisation is [0.030, 0.705], and the fixed
    triple (3.4445, -4.7750, 2.0315) is the modded-nanogpt constant for sigma_max ~ 1.

VARIANTS
  A  baseline   : Frobenius normalisation + the fixed triple, 5 steps  (production today)
  B  rescaled   : normalise so sigma_max ~ 1 (a few power iterations) + the SAME fixed triple.
                  One line. Tests whether the whole gap is just the mis-aimed input range.
  C  fitted     : Frobenius normalisation + a greedy-minimax per-step schedule fitted on
                  [0.030, 0.705] -- the Polar Express idea, fitted to our own spectra.

Scored by RMS|sigma-1| over the top-90%-energy directions, which is the quantity that survived
scrutiny. ASCII output only.
"""
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FIXED = (3.4445, -4.7750, 2.0315)
L0, U0 = 0.0297, 1.0  # sigma <= 1 is GUARANTEED by Frobenius normalisation; 0.705 is
                     # only the median max, and fitting to it let step 0 explode above it (c=66.9)
K = 5


def p_np(c, x):
    a, b, d = c
    x2 = x * x
    return x * (a + b * x2 + d * x2 * x2)


def fit_schedule(l, u, K):
    sched = []
    for _ in range(K):
        grid = np.linspace(l, u, 3000)
        def worst(c):
            v = p_np(c, grid)
            if not np.all(np.isfinite(v)):
                return 1e9
            return float(np.max(np.abs(v - 1.0)))
        best, bv = None, np.inf
        for x0 in [FIXED, (4.0, -6.0, 3.0), (5.0, -9.0, 5.0), (2.5, -2.0, 0.6), (6.0, -12.0, 7.5)]:
            r = minimize(worst, x0, method="Nelder-Mead",
                         options={"maxiter": 8000, "xatol": 1e-9, "fatol": 1e-11})
            if r.fun < bv:
                best, bv = r.x, r.fun
        c = tuple(float(v) for v in best)
        sched.append(c)
        v = p_np(c, np.linspace(l, u, 3000))
        l, u = float(v.min()), float(v.max())
    return sched


SCHED = fit_schedule(L0, U0, K)
print("fitted schedule on [%.4f, %.4f]:" % (L0, U0))
for i, c in enumerate(SCHED):
    print("   step %d: (%.6f, %.6f, %.6f)" % (i, *c))


def run(G, mode):
    X = G.to(torch.bfloat16)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.mT
    if mode == "B":
        Y = X.float()
        v = torch.randn(Y.size(1), generator=torch.Generator().manual_seed(0))
        for _ in range(12):
            v = Y.mT @ (Y @ v); v = v / (v.norm() + 1e-12)
        smax = float((Y @ v).norm())
        X = (X.float() / (smax + 1e-7)).to(torch.bfloat16)
    else:
        X = X / (X.norm() + 1e-7)
    steps = SCHED if mode == "C" else [FIXED] * K
    for (a, b, c) in steps:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.mT
    return X.float()


sd = torch.load("scratchpad/iter50_decktree/i50_d_optim_10935.pth", map_location="cpu", weights_only=False)
res = {m: [] for m in "ABC"}
for _k, v in sd["state"].items():
    mb = v.get("momentum_buffer")
    if not (torch.is_tensor(mb) and mb.ndim == 2):
        continue
    G = mb.float()
    k = min(G.shape)
    e = torch.linalg.svdvals(G)[:k] ** 2
    n90 = int((torch.cumsum(e, 0) / e.sum() < 0.90).sum()) + 1
    for mode in "ABC":
        s = torch.linalg.svdvals(run(G, mode))[:n90]
        res[mode].append(float(((s - 1) ** 2).mean().sqrt()))

print("\n  variant                                        RMS|sigma-1| on top-90%-energy dirs")
names = {"A": "A baseline (Frobenius + fixed triple)",
         "B": "B rescaled (sigma_max~1 + fixed triple)",
         "C": "C fitted schedule (Polar Express idea)"}
base = np.mean(res["A"])
for m in "ABC":
    v = np.mean(res[m])
    print(f"  {names[m]:45s} {v:.4f}   ({(v/base-1)*100:+.1f}%)")
