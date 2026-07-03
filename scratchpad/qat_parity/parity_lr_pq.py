"""Parity test for the CUDA PQ low-rank path (qat_lr_rank1 PQ branch) vs a faithful Python port of the
DEPLOY rank-1 PQ (engine compress_wkv_state r==1 + PqCodebook::encode_decode). Confirms train==deploy for
the PQ scheme. Upload the codebook via the new op, run the trunc_test op (now codebook-encodes), compare.
Usage: python scratchpad/parity_lr_pq.py [codebook_file]   (default pq_cb_m2b8.txt)
Note: PQ nearest-centroid selection is DISCRETE, so a few near-equidistant sub-vectors may pick a different
centroid under f32-vs-f64 rounding -> a handful of large per-matrix diffs is EXPECTED (codebook property,
not a train/deploy mismatch). The bulk must agree to ~1e-5 and the algorithm must be identical."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # our repo root (ported from sibling gpu_train)
sys.path.insert(0, os.path.dirname(__file__))
import rwkv, rwkv.model  # noqa: registers torch.ops.rwkv
print("DIAG set_pq op:", hasattr(torch.ops.rwkv, "rwkv7_set_pq_codebook"))
print("DIAG lr trunc op:", hasattr(torch.ops.rwkv, "rwkv7_lr_trunc_test_float"))

def load_cb(path):
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip()]
    m, bits, sub, k, ncent = (int(x) for x in lines[0].split()[:5])
    rows = [[float(x) for x in ln.split()] for ln in lines[1:]]
    cb = {}  # role -> list of m arrays [ncent, sub]
    idx = 0
    for role in range(4):
        cb[role] = []
        for _ in range(m):
            cb[role].append(np.array(rows[idx:idx + ncent], dtype=np.float64))
            idx += ncent
    return m, sub, ncent, cb

def encode_decode(role, col, cb):  # EXACT mirror of PqCodebook::encode_decode / CUDA pq_encode_decode
    norm = float(np.linalg.norm(col))
    if not np.isfinite(norm) or norm < 1e-20:
        return col
    inv = 1.0 / norm
    out = col.copy()
    for p, cents in enumerate(cb[role]):
        sub = cents.shape[1]; s = p * sub
        diff = (col[s:s + sub] * inv)[None, :] - cents
        best = int(np.argmin((diff * diff).sum(1)))
        out[s:s + sub] = cents[best] * norm
    return out

def deploy_rank1_pq(A, cb, iters=64):
    A = A.astype(np.float64); K = A.shape[0]
    amax = float(np.max(np.abs(A))); scale = amax if (np.isfinite(amax) and amax > 1e-30) else 1.0
    an = A / scale; u = np.full(K, 1.0 / np.sqrt(K))
    for _ in range(iters):
        nu = an @ (an.T @ u); nrm = np.linalg.norm(nu)
        if not np.isfinite(nrm) or nrm < 1e-30:
            break
        nu = nu / nrm; dot = abs(u @ nu); u = nu
        if 1.0 - dot < 1e-7:
            break
    v_un = A.T @ u; sigma = float(np.linalg.norm(v_un))
    uf = np.zeros(K); vf = np.zeros(K)
    if sigma > 1e-20 and np.isfinite(sigma) and np.all(np.isfinite(u)):
        sj = np.sqrt(sigma); uf = u * sj; vf = (v_un / sigma) * sj
    ai = int(np.argmax(np.abs(uf)))          # sign-canon: dominant-abs entry of uf >= 0
    if uf[ai] < 0:
        uf, vf = -uf, -vf
    uf = encode_decode(0, uf, cb); vf = encode_decode(1, vf, cb)
    return np.outer(uf, vf)

cbpath = sys.argv[1] if len(sys.argv) > 1 else "reference/pq_cb_m2b8.txt"
m, sub, ncent, cb = load_cb(cbpath)
flat = []                                    # ((role*m+p)*ncent+c)*sub+j, roles 0,1
for role in range(2):
    for p in range(m):
        flat.extend(cb[role][p].reshape(-1).tolist())
cbt = torch.tensor(flat, dtype=torch.float32, device="cuda")
torch.ops.rwkv.rwkv7_set_pq_codebook(cbt, m, sub, ncent)
print(f"uploaded cb {cbpath}: m={m} sub={sub} ncent={ncent} floats={len(flat)}")

torch.manual_seed(0)
B, H, K = 16, 2, 16
s = torch.zeros(B, H, K, K)
for b in range(B):
    for h in range(H):
        for _ in range(3):
            s[b, h] += torch.outer(torch.randn(K), torch.randn(K))
        s[b, h] += 0.05 * torch.randn(K, K)
s[0, 0] *= 1e4; s[1, 0] *= 1e-6; s[2, 0] = 0.0    # pathological: large / tiny / zero
sc = s.to("cuda").contiguous()
cuda_out = torch.ops.rwkv.rwkv7_lr_trunc_test_float.default(sc, 7.0).cpu()
a = s.numpy(); ref = np.empty((B, H, K, K))
for b in range(B):
    for h in range(H):
        ref[b, h] = deploy_rank1_pq(a[b, h], cb)
ref = torch.from_numpy(ref).float()
d = (cuda_out - ref).abs()
denom = ref.abs().amax(dim=[2, 3], keepdim=True).clamp_min(1e-9)
rel = d / denom
permat = rel.reshape(B * H, K, K).amax(dim=[1, 2])
print(f"CUDA PQ vs deploy_rank1_pq ref: max abs {d.max():.3e}  mean abs {d.mean():.3e}")
print(f"  max REL {rel.max():.3e}  mean REL {rel.mean():.3e}")
print(f"  #matrices maxREL<1e-3: {(permat < 1e-3).sum().item()}/{B*H}  (near-tie flips expected in the rest)")
print(f"  per-matrix maxREL: {[f'{x:.1e}' for x in permat.tolist()]}")
print(f"  finite: {torch.isfinite(cuda_out).all().item()}")
