#!/usr/bin/env python
"""Train the TOKEN-SHIFT PQ codebook (2 roles: 0=t_xshift `TS`, 1=c_xshift `CS`) from a
`rwkv-infer --dump-shift-corpus` corpus, and write the codebook file the engine reads via
RWKV_SHIFT_PQ (PqCodebook::load_roles(path, 2)).

Mirrors the engine's encode EXACTLY: each C-dim vector is normalized to unit; the UNIT vector is
chunked into m sub-vectors; k-means clusters each (role, position) chunk set. The norm is kept at
runtime as the scale (8 b in the accounting).

File format: line1 `m bits sub_dim C ncent`, then 2*m blocks (role-major, then pos), each `ncent`
lines of `sub_dim` floats.

Usage: python scratchpad/pq_train_shift.py <out_file> <corpus_file...> [--c 32 --m 4 --bits 8]
"""
import sys

import numpy as np


def parse_args(argv):
    out = argv[0]; files = []; o = {"c": 32, "m": 4, "bits": 8, "maxvec": 200000}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            o[a[2:]] = int(argv[i + 1]); i += 2
        else:
            files.append(a); i += 1
    return out, files, o


def load_vectors(files, c):
    roles = {0: [], 1: []}
    tags = {"TS": 0, "CS": 1}
    for f in files:
        with open(f) as fh:
            for line in fh:
                tag = line[:2]
                if tag not in tags:
                    continue
                v = np.fromstring(line[2:], sep=" ", dtype=np.float32)
                if v.size != c or not np.all(np.isfinite(v)):
                    continue
                n = float(np.linalg.norm(v))
                if n < 1e-20:
                    continue
                roles[tags[tag]].append(v / n)  # UNIT vector, like PqCodebook::encode_decode
    return roles


def main():
    out, files, o = parse_args(sys.argv[1:])
    c, m, bits = o["c"], o["m"], o["bits"]
    sub = c // m; ncent = 2 ** bits
    from sklearn.cluster import KMeans
    roles = load_vectors(files, c)
    print(f"loaded TS={len(roles[0])} CS={len(roles[1])} unit vectors; PQ m={m} bits={bits} sub={sub} ncent={ncent}")
    rng = np.random.default_rng(0)
    lines = [f"{m} {bits} {sub} {c} {ncent}"]
    for r in (0, 1):
        X = np.array(roles[r], np.float32)
        if len(X) > o["maxvec"]:
            X = X[rng.choice(len(X), o["maxvec"], replace=False)]
        for p in range(m):
            Xp = X[:, p * sub:(p + 1) * sub]
            C = KMeans(n_clusters=ncent, n_init=3, max_iter=60, random_state=0).fit(Xp).cluster_centers_
            for cc in range(ncent):
                lines.append(" ".join(f"{x:.6e}" for x in C[cc]))
            print(f"  role {r} pos {p}: {len(Xp)} vecs -> {ncent} centroids")
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}  ({len(lines)-1} centroid rows = 2 roles x {m} pos x {ncent})")


if __name__ == "__main__":
    main()
