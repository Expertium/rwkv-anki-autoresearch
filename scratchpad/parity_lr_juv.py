"""Parity test for the JOINT-UV PQ path (task23): CUDA qat_lr_rank1 joint branch vs a faithful f64
Python port of the deploy algorithm (engine compress_wkv_state r==1 + PqCodebook::encode_decode_joint).
Random joint codebook (unit-half pairs), pathological states included, two passes: norm quant OFF and
int1 over [-3,0] (the q72u deploy config). Same caveat as parity_lr_pq.py: nearest-centroid selection
is discrete -> rare near-tie flips under f32-vs-f64 are expected; the bulk must agree to ~1e-5.
NEEDS the rebuilt RWKV_CUDA (5-arg set_pq_codebook). Usage: python scratchpad/parity_lr_juv.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rwkv, rwkv.model  # noqa: registers torch.ops.rwkv

K, NCENT = 16, 1024
rng = np.random.default_rng(3)
cents = rng.normal(size=(NCENT, 2 * K)).astype(np.float64)
cents[:, :K] /= np.linalg.norm(cents[:, :K], axis=1, keepdims=True)   # unit-ish halves like k-means
cents[:, K:] /= np.linalg.norm(cents[:, K:], axis=1, keepdims=True)
cents += 0.02 * rng.normal(size=cents.shape)                           # break exact unit norm

NQ = None  # (levels, lo, hi) when norm quant on


def quant_norm(n):
    if NQ is None:
        return n
    levels, lo, hi = NQ
    t = (np.log2(n) - lo) / (hi - lo)
    q = min(max(round(t * levels), 0.0), levels)
    return float(np.exp2(lo + q / levels * (hi - lo)))


def joint_encode(uf, vf):
    nu = float(np.linalg.norm(uf)); nv = float(np.linalg.norm(vf))
    if not np.isfinite(nu) or nu < 1e-20 or not np.isfinite(nv) or nv < 1e-20:
        return uf, vf
    q = np.concatenate([uf / nu, vf / nv])
    d = ((q[None, :] - cents) ** 2).sum(1)
    best = int(np.argmin(d))
    qu, qv = quant_norm(nu), quant_norm(nv)
    return cents[best, :K] * qu, cents[best, K:] * qv


def deploy_rank1_juv(A, iters=64):
    A = A.astype(np.float64)
    amax = float(np.max(np.abs(A))); scale = amax if (np.isfinite(amax) and amax > 1e-30) else 1.0
    an = A / scale; u = np.full(K, 1.0 / np.sqrt(K))
    for _ in range(iters):
        nu_ = an @ (an.T @ u); nrm = np.linalg.norm(nu_)
        if not np.isfinite(nrm) or nrm < 1e-30:
            break
        nu_ = nu_ / nrm; dot = abs(u @ nu_); u = nu_
        if 1.0 - dot < 1e-7:
            break
    v_un = A.T @ u; sigma = float(np.linalg.norm(v_un))
    uf = np.zeros(K); vf = np.zeros(K)
    if sigma > 1e-20 and np.isfinite(sigma) and np.all(np.isfinite(u)):
        sj = np.sqrt(sigma); uf = u * sj; vf = (v_un / sigma) * sj
    ai = int(np.argmax(np.abs(uf)))
    if uf[ai] < 0:
        uf, vf = -uf, -vf
    uf, vf = joint_encode(uf, vf)
    return np.outer(uf, vf)


cbt = torch.tensor(cents.reshape(-1), dtype=torch.float32, device="cuda")
torch.ops.rwkv.rwkv7_set_pq_codebook(cbt, 1, 2 * K, NCENT, 1)   # joint=1 (5-arg = new build only)
print(f"uploaded random JOINT cb: ncent={NCENT} sub={2*K} floats={cbt.numel()}")

torch.manual_seed(0)
B, H = 16, 2
s = torch.zeros(B, H, K, K)
for b in range(B):
    for h in range(H):
        for _ in range(3):
            s[b, h] += torch.outer(torch.randn(K), torch.randn(K))
        s[b, h] += 0.05 * torch.randn(K, K)
s[0, 0] *= 1e4; s[1, 0] *= 1e-6; s[2, 0] = 0.0    # pathological: large / tiny / zero
sc = s.to("cuda").contiguous()

ok_all = True
for label, nq in (("norm-quant OFF", None), ("norm-quant int1 [-3,0]", (1.0, -3.0, 0.0))):
    NQ = nq
    torch.ops.rwkv.rwkv7_set_norm_quant(cbt, 1 if nq else 0, -3.0, 0.0)
    cuda_out = torch.ops.rwkv.rwkv7_lr_trunc_test_float.default(sc, 7.0).cpu()
    a = s.numpy(); ref = np.empty((B, H, K, K))
    for b in range(B):
        for h in range(H):
            ref[b, h] = deploy_rank1_juv(a[b, h])
    ref = torch.from_numpy(ref).float()
    d = (cuda_out - ref).abs()
    denom = ref.abs().amax(dim=[2, 3], keepdim=True).clamp_min(1e-9)
    rel = d / denom
    permat = rel.reshape(B * H, K, K).amax(dim=[1, 2])
    good = int((permat < 1e-3).sum())
    fin = bool(torch.isfinite(cuda_out).all())
    print(f"[{label}] max abs {d.max():.3e}  maxREL {rel.max():.3e}  "
          f"matrices<1e-3: {good}/{B*H}  finite: {fin}")
    ok_all &= fin and good >= B * H - 2   # allow a couple of near-tie flips
torch.ops.rwkv.rwkv7_set_norm_quant(cbt, 0, -3.0, 0.0)  # leave global state clean
print("PARITY OK" if ok_all else "PARITY FAIL")
sys.exit(0 if ok_all else 1)
