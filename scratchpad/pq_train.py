#!/usr/bin/env python
"""Train PQ codebooks for the rank-2 WKV factor directions and write a codebook file the engine reads.

Reads a corpus (`rwkv-infer --dump-corpus`), replicates the engine's rank-2 factorization + sign-canon
(must match model.rs EXACTLY so train==runtime directions), and k-means-clusters the UNIT direction
sub-vectors, per (role, sub-vec position). Roles: 0=u1, 1=v1, 2=u2, 3=v2 (rank order = dominant first).

Codebook file format (plain text, engine-parsed):
    line1:  m bits sub_dim k ncent
    then 4*m blocks (role-major, then position), each `ncent` lines of `sub_dim` floats (the centroids).

Usage: python scratchpad/pq_train.py <out_file> <corpus_file...> [--k 16 --h 2 --m 2 --bits 8]
"""
import sys, numpy as np

def parse_args(argv):
    out = argv[0]; files = []; o = {"k": 16, "h": 2, "m": 2, "bits": 8, "maxdir": 120000}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            o[a[2:]] = int(argv[i + 1]); i += 2
        else:
            files.append(a); i += 1
    return out, files, o

def load_states(files, h, k):
    hkk = h * k * k; rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.startswith("STATE"):
                    continue
                v = np.fromstring(line[5:], sep=" ", dtype=np.float32)
                if v.size == hkk and np.all(np.isfinite(v)):
                    rows.append(v)
    return np.array(rows, np.float32).reshape(-1, h, k, k)

def rank2_dirs(a, k):
    """EXACT match to model.rs::compress_wkv_state rank-2 path + sign canon. Returns [u1,v1,u2,v2] unit dirs."""
    amax = np.abs(a).max(); scale = amax if (np.isfinite(amax) and amax > 1e-30) else 1.0
    an = a / scale; gram = an @ an.T
    w, V = np.linalg.eigh(gram); order = np.argsort(w)[::-1]
    out = []
    for j in range(2):
        col = order[j]; ev = w[col]
        if not np.isfinite(ev) or ev <= 0:
            out += [np.zeros(k, np.float32), np.zeros(k, np.float32)]; continue
        sigma = np.sqrt(ev) * scale
        u = V[:, col].astype(np.float32); v = (a.T @ u / sigma).astype(np.float32)
        s = 1.0 if u[np.argmax(np.abs(u))] >= 0 else -1.0
        u, v = u * s, v * s
        for d in (u, v):
            n = np.linalg.norm(d); out.append(d / n if n > 1e-20 else d)
    return out

def collect(states, k, maxdir):
    roles = [[] for _ in range(4)]
    for st in states:
        for hh in range(st.shape[0]):
            ds = rank2_dirs(st[hh], k)
            for r in range(4):
                if np.linalg.norm(ds[r]) > 1e-6:
                    roles[r].append(ds[r])
    out = []
    rng = np.random.default_rng(0)
    for r in range(4):
        X = np.array(roles[r], np.float32)
        if len(X) > maxdir:
            X = X[rng.choice(len(X), maxdir, replace=False)]
        out.append(X)
    return out

def main():
    out, files, o = parse_args(sys.argv[1:])
    k, h, m, bits = o["k"], o["h"], o["m"], o["bits"]
    sub = k // m; ncent = 2 ** bits
    from sklearn.cluster import KMeans
    states = load_states(files, h, k)
    print(f"loaded {len(states)} states; training PQ m={m} bits={bits} sub_dim={sub} ncent={ncent}")
    dirs = collect(states, k, o["maxdir"])
    lines = [f"{m} {bits} {sub} {k} {ncent}"]
    for r in range(4):
        X = dirs[r]
        for p in range(m):
            Xp = X[:, p * sub:(p + 1) * sub]
            C = KMeans(n_clusters=ncent, n_init=3, max_iter=60, random_state=0).fit(Xp).cluster_centers_
            for c in range(ncent):
                lines.append(" ".join(f"{x:.6e}" for x in C[c]))
            print(f"  role {r} pos {p}: {len(Xp)} dirs -> {ncent} centroids")
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}  ({len(lines)-1} centroid rows = 4 roles x {m} pos x {ncent})")

if __name__ == "__main__":
    main()
