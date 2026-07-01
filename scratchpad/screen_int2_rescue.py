"""Fast numpy screen of the int2-low-rank RESCUE techniques, mirroring the Rust `lowrank_roundtrip`
math EXACTLY (Gram+eigen top-r, factors U*sqrt(s)/V*sqrt(s), per-col/Hadamard/4-level quant). Frobenius
relative reconstruction error is the cheap proxy (the rank screen already used SVD energy). This is a
DIRECTIONAL screen of the approach -- the real metric is logloss via the Rust deploy eval (run after
the build). If #3 (Hadamard) / #4 (4-level) push int2 error toward int4, the full eval is warranted.

Card WKV states are near-rank-2 (energy ~0.987 at r=2) with a long small-singular-value tail; we model
that. Run: PYTHONPATH=. .venv\\Scripts\\python.exe scratchpad/screen_int2_rescue.py
"""
import numpy as np

rng = np.random.default_rng(1234)
K, R = 32, 2
N = 400  # matrices to average over

def hadamard(k):
    h = np.array([[1.0]])
    while h.shape[0] < k:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(k)

H = hadamard(K)

def make_state():
    """Near-rank-2 KxK: two dominant singular dirs + a decaying tail (mimics real card WKV states)."""
    U = np.linalg.qr(rng.standard_normal((K, K)))[0]
    V = np.linalg.qr(rng.standard_normal((K, K)))[0]
    s = np.zeros(K)
    s[0] = rng.uniform(2.0, 6.0)
    s[1] = s[0] * rng.uniform(0.2, 0.6)            # second comp meaningfully smaller (percol matters)
    tail = np.sort(rng.uniform(0, 0.06, K - 2))[::-1] * s[1]  # small clustered tail
    s[2:] = tail
    return (U * s) @ V.T

def factors(A):
    """Top-r factors via the same sqrt-split the Rust uses: uf=U_r*sqrt(s), vf=V_r*sqrt(s)."""
    U, s, Vt = np.linalg.svd(A)
    sq = np.sqrt(s[:R])
    uf = U[:, :R] * sq            # (K,R)
    vf = Vt[:R, :].T * sq         # (K,R)
    return uf, vf

def q_int(M, qmax, per_col):
    out = M.copy()
    if per_col:
        for j in range(M.shape[1]):
            amax = np.abs(M[:, j]).max()
            sc = max(amax / qmax, 1e-12)
            out[:, j] = np.clip(np.round(M[:, j] / sc), -qmax, qmax) * sc
    else:
        amax = np.abs(M).max(); sc = max(amax / qmax, 1e-12)
        out = np.clip(np.round(M / sc), -qmax, qmax) * sc
    return out

def q_4level(M, per_col):
    out = M.copy()
    def one(col):
        amax = np.abs(col).max(); sc = max(amax / 1.5, 1e-12)
        lvl = np.clip(np.floor(col / sc), -2.0, 1.0) + 0.5   # {-1.5,-0.5,0.5,1.5}
        return lvl * sc
    if per_col:
        for j in range(M.shape[1]):
            out[:, j] = one(M[:, j])
    else:
        flat = one(M.flatten()); out = flat.reshape(M.shape)
    return out

def roundtrip(A, scheme):
    uf, vf = factors(A)
    had = scheme.get("hadamard")
    if had:
        uf, vf = H @ uf, H @ vf
    if scheme["mode"] == "fp32":
        pass
    elif scheme["mode"] == "4level":
        uf, vf = q_4level(uf, scheme["percol"]), q_4level(vf, scheme["percol"])
    else:
        qmax = scheme["qmax"]
        uf, vf = q_int(uf, qmax, scheme["percol"]), q_int(vf, qmax, scheme["percol"])
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
    ("int2 NObase (no percol)",         dict(mode="int", qmax=1, percol=False)),
]

# sanity: Hadamard with no quant must be ~identity (validates orthogonality + the un-rotate)
A0 = make_state()
uf, vf = factors(A0)
recon_plain = uf @ vf.T
recon_had = (H @ (H @ uf)) @ (H @ (H @ vf)).T
print(f"Hadamard no-quant identity check: max|diff| = {np.abs(recon_plain - recon_had).max():.2e} (want ~0)\n")

errs = {name: [] for name, _ in SCHEMES}
for _ in range(N):
    A = make_state()
    nrm = np.linalg.norm(A)
    for name, sch in SCHEMES:
        rel = np.linalg.norm(roundtrip(A, sch) - A) / nrm
        errs[name].append(rel)

print(f"Frobenius RELATIVE reconstruction error (mean over {N} near-rank-2 states), lower=better:")
base = np.mean(errs["int2 percol (the 'dies' base)"])
int4 = np.mean(errs["int4 percol (current deploy)"])
for name, _ in SCHEMES:
    m = np.mean(errs[name])
    tag = ""
    if "int2 percol +" in name:
        tag = f"  (vs int2-base x{base/m:.2f} smaller err; vs int4 {'<=' if m<=int4*1.05 else '>'} target)"
    print(f"  {name:36s} {m:.4f}{tag}")
print(f"\nTARGET = match int4 ({int4:.4f}); int2-base = {base:.4f}")
