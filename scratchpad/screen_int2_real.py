"""Re-screen the int2-rescue techniques (#1 percol, #3 Hadamard, #4 4-level) on REAL card WKV states
dumped from the Rust engine (--dump-card-state), not synthetic. This settles whether #3 (Hadamard)
helps: it only does if the real rank-2 singular vectors are COHERENT (spiky), which the synthetic
random-orthogonal screen could not exhibit. Also reports coherence mu = max|entry|*sqrt(K) of the top
singular vectors (mu~1 incoherent -> Hadamard useless; mu>>1 spiky -> Hadamard helps).

Run from repo root: PYTHONPATH=. .venv\\Scripts\\python.exe scratchpad/screen_int2_real.py
"""
import os
import re
import subprocess
import numpy as np

BIN = r"rust\rwkv-infer\target\release\rwkv-infer.exe"
W = "reference/champ_decay15.safetensors"
USERS = [107, 121, 136, 156, 162, 176]
CARD_POS = [3, 10, 40, 120]
K, R = 32, 2


def dump_matrix(user, pos):
    env = dict(os.environ, RWKV_WEIGHTS=W)
    out = subprocess.run([BIN, "--dump-card-state", str(user), str(pos)],
                         capture_output=True, text=True, env=env).stdout
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("=== fp32") and "WKV state" in ln:
            rows = []
            for r in range(i + 1, len(lines)):
                vals = lines[r].split()
                if len(vals) != K:
                    break
                try:
                    rows.append([float(x) for x in vals])
                except ValueError:
                    break
            if len(rows) == K:
                return np.array(rows)
    return None


def hadamard(k):
    h = np.array([[1.0]])
    while h.shape[0] < k:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(k)
H = hadamard(K)


def factors(A):
    U, s, Vt = np.linalg.svd(A)
    sq = np.sqrt(s[:R])
    return U[:, :R] * sq, Vt[:R, :].T * sq, U, s


def q_int(M, qmax, per_col):
    out = M.copy()
    if per_col:
        for j in range(M.shape[1]):
            sc = max(np.abs(M[:, j]).max() / qmax, 1e-12)
            out[:, j] = np.clip(np.round(M[:, j] / sc), -qmax, qmax) * sc
    else:
        sc = max(np.abs(M).max() / qmax, 1e-12)
        out = np.clip(np.round(M / sc), -qmax, qmax) * sc
    return out


def q_4level(M, per_col):
    out = M.copy()
    def one(col):
        sc = max(np.abs(col).max() / 1.5, 1e-12)
        return (np.clip(np.floor(col / sc), -2.0, 1.0) + 0.5) * sc
    if per_col:
        for j in range(M.shape[1]):
            out[:, j] = one(M[:, j])
    else:
        out = one(M.flatten()).reshape(M.shape)
    return out


def roundtrip(A, sch):
    uf, vf, _, _ = factors(A)
    had = sch.get("hadamard")
    if had:
        uf, vf = H @ uf, H @ vf
    if sch["mode"] == "fp32":
        pass
    elif sch["mode"] == "4level":
        uf, vf = q_4level(uf, sch["percol"]), q_4level(vf, sch["percol"])
    else:
        uf, vf = q_int(uf, sch["qmax"], sch["percol"]), q_int(vf, sch["qmax"], sch["percol"])
    if had:
        uf, vf = H @ uf, H @ vf
    return uf @ vf.T


SCHEMES = [
    ("rank2 fp32 (trunc floor)",        dict(mode="fp32")),
    ("int4 percol (current deploy)",    dict(mode="int", qmax=7, percol=True)),
    ("int2 percol (the 'dies' base)",   dict(mode="int", qmax=1, percol=True)),
    ("int2 percol + HADAMARD (#3)",     dict(mode="int", qmax=1, percol=True, hadamard=True)),
    ("int2 percol + 4LEVEL (#4)",       dict(mode="4level", percol=True)),
    ("int2 percol + HADAMARD + 4LEVEL", dict(mode="4level", percol=True, hadamard=True)),
]

states, coh = [], []
for u in USERS:
    for p in CARD_POS:
        A = dump_matrix(u, p)
        if A is not None and np.isfinite(A).all() and np.abs(A).max() > 1e-9:
            states.append(A)
            U, s, Vt = np.linalg.svd(A)
            # coherence of the top singular vectors (1 = perfectly spread, sqrt(K)=max spiky)
            mu = max(np.abs(U[:, 0]).max(), np.abs(Vt[0, :]).max()) * np.sqrt(K)
            coh.append(mu)
print(f"dumped {len(states)} real card states; top-singular-vector coherence mu: "
      f"mean {np.mean(coh):.2f} max {np.max(coh):.2f} (1=incoherent/Hadamard-useless, >~2 = spiky)\n")

errs = {n: [] for n, _ in SCHEMES}
for A in states:
    nrm = np.linalg.norm(A)
    if nrm < 1e-12:
        continue
    for n, s in SCHEMES:
        errs[n].append(np.linalg.norm(roundtrip(A, s) - A) / nrm)

print(f"Frobenius RELATIVE reconstruction error on REAL states (mean over {len(states)}), lower=better:")
base = np.mean(errs["int2 percol (the 'dies' base)"]); int4 = np.mean(errs["int4 percol (current deploy)"])
for n, _ in SCHEMES:
    m = np.mean(errs[n])
    tag = f"  (x{base/m:.2f} vs int2-base; {'<=int4' if m <= int4*1.10 else f'{m/int4:.1f}x int4'})" if "int2 percol +" in n else ""
    print(f"  {n:36s} {m:.4f}{tag}")
print(f"\nTARGET = int4 {int4:.4f}; int2-base {base:.4f}")
