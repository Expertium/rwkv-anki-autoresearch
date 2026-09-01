"""Do the five streams inject into the SAME subspace, or disjoint ones?

This is what actually bounds d_model, and stream_budget.py cannot answer it: five streams at
participation ratio ~7 each could jointly occupy anywhere from ~7 to ~35 dimensions. If they
share a subspace, a narrow trunk is enough; if they are disjoint, d=80 is buying something and
a naive shrink will cost.

Also reports the effective rank of the FINAL representation the heads see -- an upper bound on
everything downstream of the trunk.

Usage: .venv/Scripts/python.exe scratchpad/hybrid100k/union_rank.py
"""
import glob
import os

import numpy as np

NAMES = ["card_id", "note_id", "deck_id", "preset_id", "user_id"]


def pr_d95(X):
    X = X - X.mean(0, keepdims=True)
    lam = np.linalg.svd(X, compute_uv=False) ** 2
    if lam.sum() <= 0:
        return float("nan"), -1
    return float(lam.sum() ** 2 / (lam ** 2).sum()), int(
        np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)


def top_basis(X, k):
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k]


for f in sorted(glob.glob(os.path.join("scratchpad", "hybrid100k", "deltas_u*.npz"))):
    z = np.load(f)
    u = os.path.basename(f).split("_u")[1].split(".")[0]
    D = {n: z[n].astype(np.float64) for n in NAMES}
    x = z["x"].astype(np.float64)
    n, d = x.shape
    print("=" * 70)
    print("user %s   rows %d   d_model %d" % (u, n, d))

    pr_x, d95_x = pr_d95(x)
    print("  FINAL representation x        : eff_rank %6.2f   d95 %3d" % (pr_x, d95_x))

    # the total signal the streams inject, as one cloud
    tot = sum(D.values())
    pr_t, d95_t = pr_d95(tot)
    print("  SUM of all stream deltas      : eff_rank %6.2f   d95 %3d" % (pr_t, d95_t))

    # the union subspace: every stream's deltas as rows of one matrix
    U = np.concatenate([D[n] for n in NAMES], axis=0)
    pr_u, d95_u = pr_d95(U)
    print("  UNION (all deltas stacked)    : eff_rank %6.2f   d95 %3d" % (pr_u, d95_u))
    print("  (sum of the five per-stream d95 if fully disjoint would be ~%d)"
          % sum(pr_d95(D[nm])[1] for nm in NAMES))

    # pairwise subspace alignment: energy of stream j captured by stream i's top-k basis
    k = 8
    B = {nm: top_basis(D[nm], k) for nm in NAMES}
    print("\n  energy of ROW stream captured by COLUMN stream's top-%d basis" % k)
    print("  %-10s" % "" + "".join("%9s" % c[:6] for c in NAMES))
    for r in NAMES:
        Xr = D[r] - D[r].mean(0, keepdims=True)
        tot_e = (Xr ** 2).sum()
        cells = []
        for c in NAMES:
            proj = Xr @ B[c].T
            cells.append((proj ** 2).sum() / tot_e if tot_e > 0 else float("nan"))
        print("  %-10s" % r[:9] + "".join("%9.3f" % v for v in cells))
    print()
print("READ: if UNION eff_rank is close to the per-stream figures, the streams SHARE one")
print("      narrow subspace and a thin trunk is enough. If it approaches the disjoint sum,")
print("      d=80 is buying separation and shrinking width will cost.")
