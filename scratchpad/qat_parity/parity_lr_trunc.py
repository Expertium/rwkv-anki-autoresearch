import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # our repo root (ported from sibling gpu_train)
sys.path.insert(0, os.path.dirname(__file__))
from lr_ref import deploy_rank1_batched
import rwkv, rwkv.model
print("DIAG rwkv:", rwkv.__file__)
print("DIAG RWKV_CUDA:", getattr(rwkv.model, "RWKV_CUDA", "MISSING"))
print("DIAG qat op:", hasattr(torch.ops.rwkv, "rwkv7_wkv_qat_forward_float"))
print("DIAG lr op:", hasattr(torch.ops.rwkv, "rwkv7_lr_trunc_test_float"))
dev = "cuda"; torch.manual_seed(0)
B, H, K = 8, 2, 16
# realistic-ish states: a few rank-1s + noise; plus a couple pathological (huge, tiny, zero)
s = torch.zeros(B, H, K, K)
for b in range(B):
    for h in range(H):
        for _ in range(3):
            s[b, h] += torch.outer(torch.randn(K), torch.randn(K))
        s[b, h] += 0.05 * torch.randn(K, K)
s[0, 0] *= 1e4          # large
s[1, 0] *= 1e-6         # tiny
s[2, 0] = 0.0           # exact zero (degenerate)
sc = s.to(dev).contiguous()
cuda_out = torch.ops.rwkv.rwkv7_lr_trunc_test_float.default(sc, 7.0).cpu()
ref = deploy_rank1_batched(s, 7.0).float()
d = (cuda_out - ref).abs()
# relative to each matrix's own scale
denom = ref.abs().amax(dim=[2,3], keepdim=True).clamp_min(1e-9)
rel = (d / denom)
print(f"CUDA lr-trunc vs deploy_rank1 ref: max abs {d.max():.3e}  mean abs {d.mean():.3e}")
print(f"  max REL (per-matrix) {rel.max():.3e}  mean REL {rel.mean():.3e}")
# per-matrix frobenius agreement
cf = cuda_out.reshape(B*H,K,K).norm(dim=[1,2]); rf = ref.reshape(B*H,K,K).norm(dim=[1,2])
print(f"  frob ratio (cuda/ref) min {(cf/rf.clamp_min(1e-9)).min():.4f} max {(cf/rf.clamp_min(1e-9)).max():.4f}")
# rank check
rr = torch.linalg.matrix_rank(cuda_out.reshape(B*H,K,K))
print(f"  cuda recon ranks (nonzero mats should be 1): {rr.tolist()}")
print(f"  finite: {torch.isfinite(cuda_out).all().item()}")
