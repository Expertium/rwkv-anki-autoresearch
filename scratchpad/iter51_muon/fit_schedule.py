#!/usr/bin/env python
"""Derive a per-step Newton-Schulz coefficient schedule for OUR spectra. CPU, ~1 min.

Muon's iteration acts on each singular value independently:
    sigma  <-  p(sigma) = a*sigma + b*sigma^3 + c*sigma^5
so orthogonalization is exactly the problem "map [l, 1] onto {1}" with a degree-5 odd polynomial,
applied K times. The production code uses ONE fixed triple (3.4445, -4.7750, 2.0315) for all K
steps. That triple is optimal for a particular starting interval, and ours is not it.

This is the Polar Express idea (arXiv 2505.16932) -- per-step coefficients rather than one fixed
triple -- but the schedule here is FITTED to the singular-value range our own momentum buffers
actually occupy, and then VERIFIED against the measured error. Nothing is quoted from memory.

GREEDY MINIMAX, one step at a time: given the current interval [l, 1], choose (a,b,c) minimising
max|p(sigma) - 1| over it, then map the interval forward and repeat. Greedy is what makes it a
schedule rather than a joint optimisation, and it is what the method prescribes.

ASCII output only.
"""
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CKPT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/iter50_decktree/i50_d_optim_10935.pth"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# ---- 1) what interval do our real momentum buffers actually start in? ----
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
lo = []
for _k, v in sd["state"].items():
    mb = v.get("momentum_buffer")
    if torch.is_tensor(mb) and mb.ndim == 2:
        G = mb.float()
        s = torch.linalg.svdvals(G / (G.norm() + 1e-7))
        lo.append(float(s.min()))
lo = np.array(lo)
L0 = float(np.quantile(lo, 0.05))   # design for the 5th percentile, not the worst outlier
print(f"start sigma_min after Frobenius normalisation: median={np.median(lo):.5f} "
      f"p05={L0:.5f} min={lo.min():.5f}  (n={len(lo)})")

def p_eval(c, x):
    a, b, d = c
    x2 = x * x
    return x * (a + b * x2 + d * x2 * x2)

def fit_step(l):
    grid = np.linspace(l, 1.0, 4000)
    def worst(c):
        return np.max(np.abs(p_eval(c, grid) - 1.0))
    best, bestv = None, np.inf
    for x0 in [(3.4445, -4.7750, 2.0315), (4.0, -6.0, 3.0), (2.0, -1.5, 0.5), (8.0, -20.0, 14.0)]:
        r = minimize(worst, x0, method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12})
        if r.fun < bestv:
            best, bestv = r.x, r.fun
    return tuple(float(v) for v in best), float(bestv), grid

sched = []
l = L0
print(f"\n  {'step':>4s} {'interval lo':>12s} {'a':>10s} {'b':>10s} {'c':>10s} {'worst |p-1|':>12s}")
for k in range(K):
    c, err, grid = fit_step(l)
    vals = p_eval(c, grid)
    l_new = float(min(vals.min(), 1.0 - err))
    print(f"  {k:4d} {l:12.5f} {c[0]:10.4f} {c[1]:10.4f} {c[2]:10.4f} {err:12.5f}")
    sched.append(c)
    l = max(l_new, 1e-6)

print("\nFITTED SCHEDULE (paste into rwkv/muon.py):")
print("_POLAR_SCHEDULE = [")
for c in sched:
    print(f"    ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}),")
print("]")

# ---- baseline for comparison: the fixed triple applied K times, same scalar model ----
grid = np.linspace(L0, 1.0, 4000)
x = grid.copy()
for _ in range(K):
    x = p_eval((3.4445, -4.7750, 2.0315), x)
print(f"\nSCALAR-MODEL CHECK over [{L0:.5f}, 1]:")
print(f"  fixed triple  x{K}: RMS|s-1| = {np.sqrt(((x-1)**2).mean()):.5f}  max = {np.abs(x-1).max():.5f}")
x = grid.copy()
for c in sched:
    x = p_eval(c, x)
print(f"  fitted schedule : RMS|s-1| = {np.sqrt(((x-1)**2).mean()):.5f}  max = {np.abs(x-1).max():.5f}")
