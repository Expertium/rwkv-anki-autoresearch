#!/usr/bin/env python
"""Is the WKV codebook STALE for this trunk? A CPU-only, GPU-free pre-check.

The QAT recipe encodes rank-1 WKV factors with `reference/pq_cb_wkv_q72u.txt`
(header `1 10 32 16 1024` = joint-uv, 1024 centroids over concat(u_unit, v_unit), 2K=32 dims).
That catalog was fitted on the OLD d=32 / H=2 model. It stays dimensionally valid here because
K=16 in both -- which is precisely why the staleness is silent. This trunk is d=80 / H=5.

The question this answers, before any GPU is spent on a better codebook:

    how much of the WKV reconstruction error is the CATALOG BEING OLD,
    versus the intrinsic cost of 10 bits per head?

Method -- one corpus, one held-out split, three encoders:
  OLD     the shipped q72u catalog, encoding d=80 vectors it never saw
  REFIT   a fresh catalog fitted on the TRAIN split only, scored on the SAME held-out split
  ORACLE  a catalog fitted on the HELD-OUT split itself -- not deployable, it is the floor
          that says how much of the residual is irreducible at 1024 centroids

REFIT - OLD is what a refit buys. OLD - ORACLE bounds what ANY catalog at this budget could buy.
If REFIT ~= OLD the catalog is fine and the WKV side is not the lever; if REFIT << OLD the
shipped catalog is simply the wrong model's and a refit is nearly free.

Error metric = mean relative L2 on UNIT vectors, matching pq_train_shift.py's --holdout, so the
numbers are directly comparable to the shift-side 0.1902 / 0.1734.

Usage: python scratchpad/qat_tax/wkv_cb_staleness.py <corpus glob...> [--bits 10] [--h 5] [--k 16]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pq_train import load_states, rank2_dirs  # noqa: E402

OLD_CB = "reference/pq_cb_wkv_q72u.txt"


def load_joint_cb(path):
    """Read a joint-uv catalog: header `m bits sub c ncent`, then ncent rows of `sub` floats."""
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip()]
    m, bits, sub, c, ncent = (int(x) for x in lines[0].split()[:5])
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[1:]], dtype=np.float32)
    assert m == 1, f"{path}: expected joint-uv (m=1), got m={m}"
    assert rows.shape == (ncent, sub), f"{path}: want {(ncent, sub)} got {rows.shape}"
    return rows, bits, sub


def collect_joint(states, k):
    pairs = []
    for st in states:
        for hh in range(st.shape[0]):
            ds = rank2_dirs(st[hh], k)
            u1, v1 = ds[0], ds[1]
            if np.linalg.norm(u1) > 1e-6 and np.linalg.norm(v1) > 1e-6:
                pairs.append(np.concatenate([u1, v1]))
    return np.array(pairs, np.float32)


def encode_err(X, cb, chunk=4096):
    """Mean relative L2 error of nearest-centroid encoding. Chunked so a big corpus fits in RAM."""
    errs = []
    cb_sq = (cb ** 2).sum(1)
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        d = (xb ** 2).sum(1, keepdims=True) + cb_sq[None, :] - 2.0 * xb @ cb.T
        idx = np.argmin(d, axis=1)
        resid = xb - cb[idx]
        errs.append(np.linalg.norm(resid, axis=1) / np.maximum(np.linalg.norm(xb, axis=1), 1e-12))
    return float(np.concatenate(errs).mean())


def fit(X, ncent, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=ncent, random_state=seed, n_init=3, max_iter=100,
                         batch_size=max(4096, 3 * ncent))
    km.fit(X)
    return km.cluster_centers_.astype(np.float32)


def main():
    args = [a for a in sys.argv[1:]]
    opts = {"bits": 10, "h": 5, "k": 16, "holdout": 4000}
    files = []
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            opts[args[i][2:]] = int(args[i + 1]); i += 2
        else:
            files.extend(glob.glob(args[i])); i += 1
    if not files:
        raise SystemExit("no corpus files matched")

    h, k, ncent = opts["h"], opts["k"], 2 ** opts["bits"]
    states = load_states(files, h, k)
    print(f"loaded {len(states)} states (h={h} k={k}) from {len(files)} files")
    X = collect_joint(states, k)
    print(f"  -> {len(X)} joint (u,v) vectors, dim {X.shape[1]}")
    if len(X) < opts["holdout"] * 3:
        print(f"  ⚠ thin corpus: {len(X)} vectors for ncent={ncent} "
              f"({len(X) // max(ncent, 1)} per centroid) -- treat REFIT as a lower bound on quality")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    hold, train = X[perm[:opts["holdout"]]], X[perm[opts["holdout"]:]]

    old_cb, old_bits, old_sub = load_joint_cb(OLD_CB)
    if old_sub != X.shape[1]:
        raise SystemExit(f"catalog dim {old_sub} != corpus dim {X.shape[1]} -- wrong K, not just stale")
    print(f"\nOLD catalog {OLD_CB}: bits={old_bits} ncent={len(old_cb)} sub={old_sub}")

    e_old = encode_err(hold, old_cb)
    e_refit = encode_err(hold, fit(train, ncent))
    e_oracle = encode_err(hold, fit(hold, ncent))

    print(f"\n  OLD    (d=32-fitted, used today) {e_old:.4f}")
    print(f"  REFIT  (fitted on d=80 train)    {e_refit:.4f}   -> refit buys {e_old - e_refit:+.4f} "
          f"({100.0 * (e_old - e_refit) / max(e_old, 1e-12):+.1f}%)")
    print(f"  ORACLE (fitted on the holdout)   {e_oracle:.4f}   -> irreducible-ish floor at {ncent} centroids")
    print(f"\n  for scale, the C=80 shift refits scored 0.1902 (m2b12) / 0.1734 (m5b12) held-out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
