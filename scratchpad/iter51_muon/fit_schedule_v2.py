#!/usr/bin/env python
"""Refit the NS schedule WITH A STABILITY CONSTRAINT, and validate on BOTH early and late momentum.

WHY v2 EXISTS. v1's schedule NaN'd iter 51 at step 411, the moment warmup ended. Two errors, both
mine and both avoidable:

  1. NO STABILITY MARGIN. The greedy fit tracked intervals in exact arithmetic and let the composed
     map peak at 1.78. Newton-Schulz is only contractive while singular values stay inside the
     interval each step was fitted for; once bf16 noise pushes one past ~1, step 0's aggressive
     coefficients (7.37, -20.78, 15.19) amplify instead of contract and the iteration diverges.
     The production triple peaks at 1.20, so it carries a margin v1 threw away.
  2. WRONG DISTRIBUTION. I fitted and validated on step-10935 momentum and deployed from step 1.
     Early momentum is differently conditioned (median sigma_min 2.8e-6 vs 6.6e-5 late), and on it
     v1 produced an update 1.76e7 x baseline on one matrix -- while its MEDIAN ratio looked fine at
     0.87. A median cannot see a blow-up; only the max can.

v2 therefore (a) constrains every step's polynomial to |p(x)| <= PEAK_CAP on [0,1], with the cap
set to the production triple's own 1.20 so it is no less stable than what ships, and (b) is
scored on early AND late buffers by MAX ||O||_F ratio, not median.

ASCII output only.
"""
import io
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FIXED = (3.4445, -4.7750, 2.0315)
K = 5
PEAK_CAP = 1.20          # the production triple's own composed peak -- match it, do not exceed it
L0 = 0.0297
DOM = np.linspace(0.0, 1.0, 4000)


def p_np(c, x):
    x2 = x * x
    return x * (c[0] + c[1] * x2 + c[2] * x2 * x2)


def fit(l, u):
    grid = np.linspace(l, u, 3000)
    def obj(c):
        v = p_np(c, grid)
        if not np.all(np.isfinite(v)):
            return 1e9
        full = p_np(c, DOM)
        pen = max(0.0, float(np.abs(full).max()) - PEAK_CAP)
        return float(np.max(np.abs(v - 1.0))) + 50.0 * pen
    best, bv = None, np.inf
    for x0 in [FIXED, (3.0, -3.0, 1.0), (2.5, -2.0, 0.6), (4.0, -6.0, 3.0)]:
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 12000, "xatol": 1e-10, "fatol": 1e-12})
        if r.fun < bv:
            best, bv = r.x, r.fun
    return tuple(float(v) for v in best)


sched, l, u = [], L0, 1.0
for _ in range(K):
    c = fit(l, u)
    sched.append(c)
    v = p_np(c, np.linspace(l, u, 3000))
    l, u = float(v.min()), float(v.max())

x = DOM.copy()
peak = 0.0
for c in sched:
    x = p_np(c, x)
    peak = max(peak, float(np.abs(x).max()))
xf = DOM.copy()
pf = 0.0
for _ in range(K):
    xf = p_np(FIXED, xf)
    pf = max(pf, float(np.abs(xf).max()))

print("v2 schedule (peak cap %.2f):" % PEAK_CAP)
for i, c in enumerate(sched):
    print("   step %d: (%.6f, %.6f, %.6f)" % (i, *c))
print("\n  composed peak over [0,1]:  v2 %.3f   production %.3f" % (peak, pf))


def run(G, steps):
    X = G.to(torch.bfloat16)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for a, b, c in steps:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return (X.mT if tr else X).float()


print("\n  VALIDATION -- max matters, not median (a median cannot see a blow-up)")
for tag, path in (("EARLY (step ~410)", "scratchpad/iter51_muon/i51_ws_optim_1000.pth"),
                  ("LATE  (step 10935)", "scratchpad/iter50_decktree/i50_d_optim_10935.pth")):
    if not os.path.exists(path):
        print(f"  {tag}: missing"); continue
    sd = torch.load(path, map_location="cpu", weights_only=False)
    G = [v["momentum_buffer"].float() for v in sd["state"].values()
         if torch.is_tensor(v.get("momentum_buffer")) and v["momentum_buffer"].ndim == 2]
    ratio, err_new, err_old, nf = [], [], [], 0
    for g in G:
        o_n, o_f = run(g, sched), run(g, [FIXED] * K)
        if not torch.isfinite(o_n).all():
            nf += 1
        ratio.append(float(o_n.norm() / (o_f.norm() + 1e-12)))
        k = min(g.shape)
        e = torch.linalg.svdvals(g)[:k] ** 2
        n90 = int((torch.cumsum(e, 0) / e.sum() < 0.90).sum()) + 1
        for o, acc in ((o_n, err_new), (o_f, err_old)):
            s = torch.linalg.svdvals(o)[:n90]
            acc.append(float(((s - 1) ** 2).mean().sqrt()))
    print(f"  {tag}: ||O||_F ratio median {np.median(ratio):.4f}  MAX {np.max(ratio):.4f}  "
          f"non-finite {nf}/{len(G)}")
    print(f"     top-90%-energy RMS|s-1|: production {np.mean(err_old):.4f} -> v2 {np.mean(err_new):.4f}")
