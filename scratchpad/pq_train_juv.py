#!/usr/bin/env python
"""Train the JOINT-UV WKV codebook (task23): one catalog of concat(u_unit, v_unit) 2K-dim entries,
ONE index per head selects BOTH rank-1 factor directions — the m2b12 "index bits != catalog size"
principle applied to the WKV side (same 20 WKV bits/card at bits=10, 32x the catalog, u/v correlation
captured). Reuses pq_train.py's corpus loader + EXACT engine-matching rank-1 factor extraction
(rank2_dirs first pair = u1,v1 incl. sign canon); k-means over the PAIRED concat vectors.

Header: `1 <bits> 2K K ncent` (sub_dim == 2*K + m == 1 => engine/QAT auto-detect joint mode), then
ONE block of ncent rows of 2K floats.

Usage: python scratchpad/pq_train_juv.py <out_file> <corpus_file...> [--k 16 --h 2 --bits 10]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pq_train import load_states, parse_args, rank2_dirs


def collect_joint(states, k, maxdir):
    pairs = []
    for st in states:
        for hh in range(st.shape[0]):
            ds = rank2_dirs(st[hh], k)
            u1, v1 = ds[0], ds[1]
            if np.linalg.norm(u1) > 1e-6 and np.linalg.norm(v1) > 1e-6:
                pairs.append(np.concatenate([u1, v1]))
    X = np.array(pairs, np.float32)
    rng = np.random.default_rng(0)
    if len(X) > maxdir:
        X = X[rng.choice(len(X), maxdir, replace=False)]
    return X


def main():
    out, files, o = parse_args(sys.argv[1:])
    k, h, bits = o["k"], o["h"], o["bits"]
    ncent = 2 ** bits
    from sklearn.cluster import KMeans
    states = load_states(files, h, k)
    print(f"loaded {len(states)} states; training JOINT-UV cb bits={bits} sub_dim={2*k} ncent={ncent}")
    X = collect_joint(states, k, o["maxdir"])
    print(f"  {len(X)} (u,v) pairs -> k-means k={ncent} (~{len(X)//max(ncent,1)} pairs/centroid)")
    # 2026-09-20: full KMeans is O(pairs x ncent) per iteration -- fine at 1,024 centroids, hopeless at
    # 8,192-32,768 (the top-down bit sweep). MiniBatchKMeans is what pq_train_shift.py already uses for
    # its 4,096-entry catalogs. Auto above 2,048 centroids; --minibatch 0 forces the exact fit.
    mb = o.get("minibatch", 1 if ncent > 2048 else 0)

    def fit_one(Z, nc, seed=0):
        if mb:
            from sklearn.cluster import MiniBatchKMeans
            return MiniBatchKMeans(n_clusters=nc, batch_size=max(8192, 4 * nc), n_init=3, max_iter=300,
                                   max_no_improvement=30, random_state=seed).fit(Z).cluster_centers_
        return KMeans(n_clusters=nc, n_init=3, max_iter=60, random_state=seed).fit(Z).cluster_centers_

    def _assign(Z, Cc, chunk=4096):
        idx = np.empty(len(Z), np.int64)
        for i in range(0, len(Z), chunk):
            z = Z[i:i + chunk]
            idx[i:i + chunk] = ((Cc[None, :, :] - z[:, None, :]) ** 2).sum(-1).argmin(1)
        return idx

    C = fit_one(X, ncent, 0)
    # RESIDUAL VQ (2026-09-20, Andrew's +100%): stage 2 codes what stage 1 left. Two 15-bit indices cost
    # 30 bits/head and search 65,536 centroids, but address 2^30 points -- and both stages are FITTABLE,
    # which a single b30 catalog never could be. Measured on the held-out user (sim_rvq.py, 832k training
    # pairs): 1 x b10 0.1697, 2 x b10 0.1100, 2 x b13 0.0708, 3 x b10 0.0779 -- fewer, bigger stages win.
    # Header gains a 6th field; a 5-field file is stages=1 and parses exactly as before.
    stages = o.get("stages", 1)
    blocks = [C]
    R = X
    for st in range(2, stages + 1):
        R = R - blocks[-1][_assign(R, blocks[-1])]
        print(f"  stage {st}: residual rms {float(np.sqrt((R ** 2).sum(1).mean())):.4f} -> k-means")
        blocks.append(fit_one(R, ncent, seed=st))
    head = f"1 {bits} {2*k} {k} {ncent}" + (f" {stages}" if stages > 1 else "")
    lines = [head]
    for B in blocks:
        for c in range(ncent):
            lines.append(" ".join(f"{x:.6e}" for x in B[c]))
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}  ({stages} stage(s) x {ncent} joint centroid rows of {2*k} floats)")


if __name__ == "__main__":
    main()
