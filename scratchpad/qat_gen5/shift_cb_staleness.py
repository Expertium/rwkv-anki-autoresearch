"""Is the deployed TOKEN-SHIFT catalog stale for the gen-5 trunk? Same question as
wkv_cb_staleness.py, on the other half of the QAT state budget.

The WKV side dominates the QAT tax (2026-08-12: whole shift cost +0.000365/+0.000720 against
+0.003235/+0.004183 for the WKV swap), so this exists to bound the shift side rather than to
find a large win. It mirrors the engine's encode exactly: normalise each C-dim vector to unit,
chunk the UNIT vector into m sub-vectors, encode each chunk against its own (role, pos) block.

RANDOM is the control that matters: a catalog aimed at the wrong subspace passes every shape
check and fails silently, so "beats random" is the floor a deployed catalog must clear.

Usage: shift_cb_staleness.py <catalog> <corpus glob...> [--holdout-user 156]
"""

import glob
import sys

import numpy as np


def load_cb(path):
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip()]
    m, bits, sub, c, ncent = (int(x) for x in lines[0].split()[:5])
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[1:]], np.float32)
    assert rows.shape == (2 * m * ncent, sub), f"{path}: want {(2*m*ncent, sub)} got {rows.shape}"
    return rows.reshape(2, m, ncent, sub), m, bits, sub, c, ncent


def load_vecs(files, c):
    ts, cs = [], []
    for f in files:
        for ln in open(f):
            parts = ln.split()
            if len(parts) != c + 1:
                continue
            v = np.array(parts[1:], np.float32)
            (ts if parts[0] == "TS" else cs).append(v)
    return np.array(ts), np.array(cs)


def rel_err(X, blocks, m, sub):
    """Mean relative L2 of the engine's encode: unit-normalise, chunk, nearest centroid per chunk."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / np.maximum(n, 1e-12)
    rec = np.empty_like(U)
    for j in range(m):
        xb = U[:, j * sub:(j + 1) * sub]
        cb = blocks[j]
        d = (xb ** 2).sum(1, keepdims=True) + (cb ** 2).sum(1)[None, :] - 2.0 * xb @ cb.T
        rec[:, j * sub:(j + 1) * sub] = cb[np.argmin(d, axis=1)]
    return float((np.linalg.norm(U - rec, axis=1) / np.maximum(np.linalg.norm(U, axis=1), 1e-12)).mean())


def main():
    cat = sys.argv[1]
    args, hold_user = sys.argv[2:], None
    if "--holdout-user" in args:
        i = args.index("--holdout-user")
        hold_user = args[i + 1]
        args = args[:i] + args[i + 2:]
    files = [f for a in args for f in glob.glob(a)]
    if not files:
        raise SystemExit("no corpus files matched")
    cb, m, bits, sub, c, ncent = load_cb(cat)
    print(f"catalog {cat}: m={m} bits={bits} sub={sub} C={c} ncent={ncent}")

    hold_files = [f for f in files if hold_user and f"_{hold_user}_" in f]
    train_files = [f for f in files if f not in hold_files]
    if hold_user and not hold_files:
        raise SystemExit(f"--holdout-user {hold_user} matched no files")
    use = hold_files or files
    print(f"  scoring on {len(use)} file(s){' (held out)' if hold_files else ''}, "
          f"train {len(train_files)}")

    rng = np.random.default_rng(0)
    for role, name in ((0, "TS (time-mix shift)"), (1, "CS (chan-mix shift)")):
        X = load_vecs(use, c)[role]
        if len(X) == 0:
            continue
        e_old = rel_err(X, cb[role], m, sub)
        rnd = rng.standard_normal((m, ncent, sub)).astype(np.float32)
        rnd /= np.linalg.norm(rnd, axis=2, keepdims=True)
        rnd *= float(np.sqrt(1.0 / m))  # a unit vector's chunk has expected norm 1/sqrt(m)
        e_rand = rel_err(X, rnd, m, sub)
        print(f"  {name:<22} n={len(X):>7,}  DEPLOYED {e_old:.4f}   RANDOM {e_rand:.4f}"
              f"   -> {'STALE' if e_old > 0.9 * e_rand else 'ok'}")


if __name__ == "__main__":
    main()
