"""Faithful Python port of the DEPLOY rank-1 int4 compression (engine/src/model.rs::compress_wkv_state,
r==1 fast path + quant_factor_percol_inplace). This is the train==deploy parity target for the fused
low-rank QAT kernel. All math in float64 like the Rust f64 quant; HALF-AWAY rounding (Rust f64::round).
"""
import numpy as np
import torch

def _round_half_away(x):  # Rust f64::round / CUDA roundf: ties away from zero
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def _qcol_int4(c, qmax=7.0):  # per-column symmetric int-N (quant_factor_percol_inplace)
    a = float(np.max(np.abs(c))) if c.size else 0.0
    s = max(a / qmax, 1e-12)
    return _round_half_away(c / s).clip(-qmax, qmax) * s

def deploy_rank1(A, qmax=7.0, iters=64):
    """A: (K,K) float. Returns rank-1 int4 reconstruction (K,K), matching the deploy engine."""
    A = A.astype(np.float64)
    K = A.shape[0]
    amax = float(np.max(np.abs(A)))
    scale = amax if (np.isfinite(amax) and amax > 1e-30) else 1.0
    an = A / scale
    u = np.full(K, 1.0 / np.sqrt(K))
    for _ in range(iters):
        atu = an.T @ u
        nu = an @ atu
        nrm = np.linalg.norm(nu)
        if not np.isfinite(nrm) or nrm < 1e-30:
            break
        nu = nu / nrm
        dot = abs(u @ nu)
        u = nu
        if 1.0 - dot < 1e-7:
            break
    v_un = A.T @ u  # original A -> true sigma
    sigma = float(np.linalg.norm(v_un))
    uf = np.zeros(K); vf = np.zeros(K)
    if sigma > 1e-20 and np.isfinite(sigma) and np.all(np.isfinite(u)):
        sj = np.sqrt(sigma)
        uf = u * sj
        vf = (v_un / sigma) * sj
    uf = _qcol_int4(uf, qmax)
    vf = _qcol_int4(vf, qmax)
    return np.outer(uf, vf)  # (K,K)

def deploy_rank1_batched(s_BHKK, qmax=7.0):
    B, H, K, _ = s_BHKK.shape
    a = s_BHKK.detach().cpu().numpy().astype(np.float64)
    out = np.empty_like(a, dtype=np.float64)
    for b in range(B):
        for h in range(H):
            out[b, h] = deploy_rank1(a[b, h], qmax)
    return torch.from_numpy(out)

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_train"))
    from rwkv.model.rwkv_ops import fake_lowrank_state
    torch.manual_seed(0)
    # realistic near-low-rank-ish states: sum of a couple rank-1s + noise
    B, H, K = 6, 2, 16
    s = torch.zeros(B, H, K, K)
    for b in range(B):
        for h in range(H):
            for _ in range(3):
                s[b, h] += torch.outer(torch.randn(K), torch.randn(K))
            s[b, h] += 0.05 * torch.randn(K, K)
    ref = deploy_rank1_batched(s, 7.0).float()
    fl = fake_lowrank_state(s.cuda(), 1, 7.0).cpu() if torch.cuda.is_available() else fake_lowrank_state(s, 1, 7.0)
    d = (ref - fl).abs()
    print(f"deploy_rank1 vs fake_lowrank_state(rank1,int4): max {d.max():.4e} mean {d.mean():.4e}")
    print(f"  ref frob {ref.norm():.3f}  fake_lr frob {fl.norm():.3f}  state frob {s.norm():.3f}")
    # rank check
    rr = torch.linalg.matrix_rank(ref.reshape(B*H,K,K))
    print(f"  recon ranks (should all be 1): {rr.tolist()[:8]}")
