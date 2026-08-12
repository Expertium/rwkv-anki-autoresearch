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
Optional (added 2026-08-12, all default to the ORIGINAL behaviour so the shipped catalogs stay
reproducible): --ninit/--maxiter tune the k-means budget, --minibatch 1 swaps in
MiniBatchKMeans (needed at ncent=4096: full Lloyd is hours per chunk), --holdout N keeps N
vectors per role out of the fit and reports mean relative L2 reconstruction error on them.
"""
import sys

import numpy as np


def parse_args(argv):
    out = argv[0]; files = []; o = {"c": 32, "m": 4, "bits": 8, "maxvec": 200000,
                                   "ninit": 3, "maxiter": 60, "minibatch": 0, "holdout": 0}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            o[a[2:]] = int(argv[i + 1]); i += 2
        else:
            files.append(a); i += 1
    # cmd.exe does NOT expand wildcards (unlike a POSIX shell), so a caller passing
    # `corpus\shift_*.txt` would hand us the literal string. Expand here.
    import glob as _glob
    expanded = []
    for f in files:
        hits = sorted(_glob.glob(f))
        expanded.extend(hits if hits else [f])
    return out, expanded, o


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
    from sklearn.cluster import KMeans, MiniBatchKMeans
    roles = load_vectors(files, c)
    print(f"loaded TS={len(roles[0])} CS={len(roles[1])} unit vectors; PQ m={m} bits={bits} sub={sub} ncent={ncent}")
    rng = np.random.default_rng(0)
    lines = [f"{m} {bits} {sub} {c} {ncent}"]
    # --holdout N: keep N vectors per role OUT of the fit and report reconstruction error on them.
    # Fitting error alone is meaningless at ncent=4096 (the catalog can memorize); held-out error is
    # what says whether a bit budget actually captures this model's shift distribution.
    err = {}
    for r in (0, 1):
        X = np.array(roles[r], np.float32)
        rng.shuffle(X)
        hold = X[:o["holdout"]] if o["holdout"] else None
        X = X[o["holdout"]:]
        if len(X) > o["maxvec"]:
            X = X[rng.choice(len(X), o["maxvec"], replace=False)]
        sq = np.zeros(len(hold)) if hold is not None else None
        for p in range(m):
            Xp = X[:, p * sub:(p + 1) * sub]
            if o["minibatch"]:
                km = MiniBatchKMeans(n_clusters=ncent, n_init=o["ninit"], max_iter=o["maxiter"],
                                     batch_size=min(len(Xp), 8192), random_state=0).fit(Xp)
            else:
                km = KMeans(n_clusters=ncent, n_init=o["ninit"], max_iter=o["maxiter"],
                            random_state=0).fit(Xp)
            C = km.cluster_centers_
            for cc in range(ncent):
                lines.append(" ".join(f"{x:.6e}" for x in C[cc]))
            if hold is not None:
                Hp = hold[:, p * sub:(p + 1) * sub]
                d = ((Hp[:, None, :] - C[None, :, :]) ** 2).sum(-1) if len(Hp) * ncent < 4_000_000 else None
                if d is None:   # chunk to bound memory
                    best = np.empty(len(Hp))
                    for i0 in range(0, len(Hp), 256):
                        blk = Hp[i0:i0 + 256]
                        best[i0:i0 + 256] = ((blk[:, None, :] - C[None, :, :]) ** 2).sum(-1).min(1)
                else:
                    best = d.min(1)
                sq += best
            print(f"  role {r} pos {p}: {len(Xp)} vecs -> {ncent} centroids")
        if hold is not None:
            # vectors are UNIT, so ||x||=1 and relative L2 error = sqrt(sum of per-chunk sq error)
            err[r] = float(np.sqrt(sq).mean())
            print(f"  role {r} HELD-OUT mean relative L2 error on {len(hold)} vecs: {err[r]:.4f}")
    if err:
        print(f"HOLDOUT_ERR m={m} bits={bits} TS={err.get(0, float('nan')):.4f} "
              f"CS={err.get(1, float('nan')):.4f}")
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}  ({len(lines)-1} centroid rows = 2 roles x {m} pos x {ncent})")


if __name__ == "__main__":
    main()
