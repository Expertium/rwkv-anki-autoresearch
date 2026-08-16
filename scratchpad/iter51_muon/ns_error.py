#!/usr/bin/env python
"""How far from orthogonal is our Newton-Schulz output, on REAL momentum buffers? CPU, seconds.

PROPOSALS.md #5 claims "the Newton-Schulz orthogonality error was measured real (0.19-0.31 RMS)"
and that is the entire motivation for a Polar-Express-style refinement. Reproduce it before
building anything -- the measurement is 3 months old and the trunk has changed twice since.

WHAT ORTHOGONALITY MEANS HERE. Muon wants the polar factor UV^T of the momentum G = U S V^T, i.e.
a matrix whose singular values are ALL EXACTLY 1. Newton-Schulz approximates it, so the error is
the deviation of the output's singular values from 1. Reported as RMS(|sigma - 1|) over the
min(m,n) singular values, which is directly comparable across shapes.

⚠ Two things that would fool this measurement:
  * bf16. The production path runs the iteration in bfloat16, so part of any error is precision,
    not the polynomial. Measured BOTH ways to separate them.
  * Rank deficiency. A momentum matrix with true zero singular values can never be mapped to all
    ones, so those directions are error by construction, not by the iteration's fault. Reported
    alongside the effective rank.

ASCII output only.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv.muon import zeropower_via_newtonschulz5  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/iter50_decktree/i50_d_optim_10935.pth"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
bufs = []
for k, v in sd["state"].items():
    mb = v.get("momentum_buffer")
    if torch.is_tensor(mb) and mb.ndim == 2:
        bufs.append((k, mb.float()))
print(f"{len(bufs)} momentum matrices from {CKPT}, ns_steps={STEPS}")

def sv_err(X, k):
    s = torch.linalg.svdvals(X.float())[:k]
    return float(((s - 1.0) ** 2).mean().sqrt()), float(s.min()), float(s.max())

shapes = {}
for k, G in bufs:
    shapes.setdefault(tuple(G.shape), []).append((k, G))

print(f"\n  {'shape':>12s} {'n':>3s} {'rank/min(m,n)':>14s} {'RMS|s-1| bf16':>14s} {'RMS|s-1| fp32':>14s}")
tot_bf, tot_fp, n_tot = 0.0, 0.0, 0
for shp, items in sorted(shapes.items(), key=lambda x: -len(x[1])):
    eb, ef, rk = [], [], []
    for _k, G in items[:12]:
        k = min(G.shape)
        s0 = torch.linalg.svdvals(G)
        rk.append(float((s0 > s0.max() * 1e-6).sum()) / k)
        eb.append(sv_err(zeropower_via_newtonschulz5(G, steps=STEPS), k)[0])
        # fp32 reference run of the identical polynomial
        X = G.clone()
        tr = X.size(0) > X.size(1)
        if tr:
            X = X.mT
        X = X / (X.norm() + 1e-7)
        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(STEPS):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if tr:
            X = X.mT
        ef.append(sv_err(X, k)[0])
    m = lambda z: sum(z) / len(z)
    print(f"  {str(shp):>12s} {len(items):3d} {m(rk):14.3f} {m(eb):14.4f} {m(ef):14.4f}")
    tot_bf += m(eb) * len(items); tot_fp += m(ef) * len(items); n_tot += len(items)
print(f"\n  WEIGHTED MEAN  bf16={tot_bf/n_tot:.4f}   fp32={tot_fp/n_tot:.4f}")
print("  (0.0 would be a perfect polar factor; PROPOSALS.md #5 claims 0.19-0.31)")
