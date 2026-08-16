#!/usr/bin/env python
"""Is Muon's NS STEP COUNT a live dose knob for the spectral flattening? CPU, ~1 min.

WHY THIS AND NOT PolarExpress. Iter 51 established that the production triple's contraction
(p(1) = a+b+c = 0.7010 < 1) is a STABILITY GUARANTEE, so any schedule that pushes p(1) -> 1
diverges on thin rank-1 momentum. Iterating the SAME triple more times keeps that contraction at
every step, so it cannot hit that failure mode. And per the 2026-08-16 regime measurement, what
Muon buys us is REGULARIZATION (spectral flattening of the update), not descent speed -- so the
step count is a dose knob on the half that actually pays, unlike PolarExpress/NorMuon which
refine descent quality.

THE SCREEN. If going 5 -> 8 steps barely moves the singular-value distribution, the lever is dead
before any GPU is spent. If it moves it materially, it is a candidate (one integer, no new math,
training-only, no deploy debt).

⚠ ITER 51'S TWO LESSONS ARE BUILT IN:
  * measured on EARLY (step 50) *and* LATE (step 10935) momentum, because early buffers are
    differently conditioned (median sigma_min 2.8e-6 vs 6.6e-5) and that is what killed iter 51;
  * every stability ratio is reported as a MAX, never a median -- a median cannot see a blow-up.

ASCII output only.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TRIPLE = (3.4445, -4.7750, 2.0315)
STEP_GRID = [1, 2, 3, 5, 8, 12, 20]
REF = 5  # the production step count everything is compared against


def ns(G, steps):
    """The production iteration, fp32, `steps` applications of the SAME triple."""
    X = G.clone().float()
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    a, b, c = TRIPLE
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if tr else X


def load_bufs(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    out = []
    for k, v in sd["state"].items():
        mb = v.get("momentum_buffer")
        if torch.is_tensor(mb) and mb.ndim == 2:
            out.append((k, mb.float()))
    return out


def report(tag, ckpt):
    bufs = load_bufs(ckpt)
    if not bufs:
        print(f"\n{tag}: no 2D momentum buffers in {ckpt} -- SKIPPED")
        return
    print(f"\n=== {tag}: {len(bufs)} momentum matrices from {ckpt} ===")

    # pre-compute each buffer's own spectrum once (for the top-90%-energy mask)
    pre = []
    for k, G in bufs:
        s0 = torch.linalg.svdvals(G)
        e = torch.cumsum(s0 ** 2, 0) / (s0 ** 2).sum()
        n90 = int((e < 0.90).sum()) + 1  # directions carrying the top 90% of energy
        pre.append((k, G, min(G.shape), n90))

    ref_norm = {k: ns(G, REF).norm().item() for k, G, _, _ in pre}
    print(f"  {'steps':>6s} {'RMS|s-1| all':>13s} {'RMS|s-1| top90':>15s} "
          f"{'MAX |O|F / ref':>15s} {'MAX sigma':>10s}")
    for st in STEP_GRID:
        all_e, top_e, ratios, smax = [], [], [], []
        for k, G, kk, n90 in pre:
            O = ns(G, st)
            s = torch.linalg.svdvals(O)[:kk]
            all_e.append(float(((s - 1.0) ** 2).mean().sqrt()))
            top_e.append(float(((s[:n90] - 1.0) ** 2).mean().sqrt()))
            ratios.append(O.norm().item() / max(ref_norm[k], 1e-12))
            smax.append(float(s.max()))
        m = lambda z: sum(z) / len(z)
        mark = "  <- production" if st == REF else ""
        print(f"  {st:6d} {m(all_e):13.4f} {m(top_e):15.4f} "
              f"{max(ratios):15.4f} {max(smax):10.4f}{mark}")


def main():
    late = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/iter50_decktree/i50_d_optim_10935.pth"
    early = sys.argv[2] if len(sys.argv) > 2 else "scratchpad/iter51_muon/i51_ws_optim_50.pth"
    for tag, c in (("EARLY (step 50)", early), ("LATE (step 10935)", late)):
        if os.path.exists(c):
            report(tag, c)
        else:
            print(f"\n{tag}: MISSING {c}")
    print("\nREAD IT AS: if 'top90' is already flat by step 5 and 8/12 barely improve it, the")
    print("dose knob is saturated and the lever is dead. MAX |O|F/ref and MAX sigma are the")
    print("stability guards -- iterating a contraction should keep both bounded (iter 51).")


if __name__ == "__main__":
    main()
