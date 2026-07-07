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
    C = KMeans(n_clusters=ncent, n_init=3, max_iter=60, random_state=0).fit(X).cluster_centers_
    lines = [f"1 {bits} {2*k} {k} {ncent}"]
    for c in range(ncent):
        lines.append(" ".join(f"{x:.6e}" for x in C[c]))
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}  ({ncent} joint centroid rows of {2*k} floats)")


if __name__ == "__main__":
    main()
