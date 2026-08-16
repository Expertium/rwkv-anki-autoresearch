#!/usr/bin/env python
"""Is the Newton-Schulz error in directions that MATTER, or in the near-null tail? CPU, seconds.

WHY THIS EXISTS. PROPOSALS.md #5 is motivated entirely by "orthogonality error 0.19-0.31 RMS",
which reproduces. But RMS over ALL singular values weights a direction carrying 1e-10 of the
momentum's energy exactly as much as the top one. Our buffers turn out to have sigma_min/||G||_F
~ 7e-5 (median), so the interval is effectively [0, 1] and NO odd polynomial can map 0 -> 1 in
five steps. Part of that 0.29 is therefore irreducible by construction -- and arguably SHOULD not
be reduced, since lifting a near-null direction to 1 amplifies pure noise.

So the decision-relevant number is the error restricted to the directions that carry the update:
the top-k singular values holding 90 / 99 / 99.9 percent of the spectrum's energy.

This is the iter-47 lesson applied BEFORE spending a run rather than after: a proxy that can
diverge from the quantity you care about has to be checked against that quantity first.

ASCII output only.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv.muon import zeropower_via_newtonschulz5  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/iter50_decktree/i50_d_optim_10935.pth"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)

rows = []
for _k, v in sd["state"].items():
    mb = v.get("momentum_buffer")
    if torch.is_tensor(mb) and mb.ndim == 2:
        G = mb.float()
        s_in = torch.linalg.svdvals(G)
        X = zeropower_via_newtonschulz5(G, steps=5).float()
        s_out = torch.linalg.svdvals(X)
        k = min(G.shape)
        e2 = (s_out[:k] - 1.0) ** 2
        energy = (s_in[:k] ** 2)
        cum = torch.cumsum(energy, 0) / energy.sum()
        row = {"all": float(e2.mean().sqrt()), "cond": float(s_in[0] / s_in[k - 1].clamp_min(1e-30))}
        for tag, q in (("90", 0.90), ("99", 0.99), ("999", 0.999)):
            n = int((cum < q).sum()) + 1
            row[tag] = float(e2[:n].mean().sqrt())
            row["n" + tag] = n
        row["k"] = k
        rows.append(row)

m = lambda key: sum(r[key] for r in rows) / len(rows)
print(f"{len(rows)} momentum matrices, ns_steps=5\n")
print(f"  RMS|sigma-1| over ALL singular values          : {m('all'):.4f}   <- what #5 quotes")
print(f"  RMS|sigma-1| over the top 90% of energy        : {m('90'):.4f}   (mean {m('n90'):.1f} of {m('k'):.1f} dirs)")
print(f"  RMS|sigma-1| over the top 99% of energy        : {m('99'):.4f}   (mean {m('n99'):.1f} dirs)")
print(f"  RMS|sigma-1| over the top 99.9% of energy      : {m('999'):.4f}   (mean {m('n999'):.1f} dirs)")
print(f"\n  median condition number of the momentum       : {sorted(r['cond'] for r in rows)[len(rows)//2]:.3e}")
print("\n  If the error collapses on the energy-bearing directions, the 0.29 is the near-null")
print("  tail -- irreducible in 5 steps, and undesirable to reduce -- and #5's motivation is gone.")
