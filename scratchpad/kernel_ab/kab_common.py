"""Shared deterministic WKV input generator for the Tier-1 kernel A/B (cudaMalloc -> torch::empty).
Inputs are drawn on CPU with a fixed-seed generator (stable across processes), then moved to CUDA, so
the golden (production .pyd) and test (build/ .pyd) processes see byte-identical inputs. Imports ONLY
torch, so the test process can use it WITHOUT importing rwkv (which would load the locked production
.pyd). Shapes default to the champion regime (H=2, K=16) with T long enough to trigger BOTH
time-parallel paths (fwd needs T>=3*512=1536, bwd needs T>=3*128=384) -- the only paths the change
touches."""
import torch


def make_inputs(B=8, T=1600, H=2, K=16, dtype=torch.float32, device="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)  # CPU generator: identical draw order across processes

    def rn(shape, scale):
        return (torch.randn(shape, generator=g, dtype=torch.float32) * scale)

    r = rn((B, T, H, K), 0.5).to(device=device, dtype=dtype).contiguous()
    k = rn((B, T, H, K), 0.5).to(device=device, dtype=dtype).contiguous()
    v = rn((B, T, H, K), 0.5).to(device=device, dtype=dtype).contiguous()
    a = rn((B, T, H, K), 0.1).to(device=device, dtype=dtype).contiguous()
    k_deformed = rn((B, T, H, K), 0.1).to(device=device, dtype=dtype).contiguous()
    # w is the per-step decay; must be in (0,1) for a stable recurrence. float32 always.
    w = (torch.rand((B, T, H, K), generator=g) * 0.1 + 0.85).to(device=device, dtype=torch.float32).contiguous()
    skip = torch.zeros((B, T), dtype=torch.bool, device=device)
    grad = rn((B, T, H, K), 0.5).to(device=device, dtype=dtype).contiguous()
    return r, k, v, w, a, k_deformed, skip, grad


def run_fwd_bwd(r, k, v, w, a, k_deformed, skip, grad):
    """One forward + backward through the float32 WKV op. Returns (out, ckpt, [6 grads])."""
    out, ckpt = torch.ops.rwkv.rwkv7_wkv_forward_float.default(r, k, v, w, a, k_deformed, skip)
    grads = torch.ops.rwkv.rwkv7_wkv_backward_float.default(r, k, v, w, a, k_deformed, skip, ckpt, grad)
    return out, ckpt, list(grads)
