"""Op-level parity test for the stateful-BPTT WKV kernel (RWKV7_WKV_Stateful).

Three checks, each for float32 AND bfloat16:
  (A) state0=zeros equals the existing non-stateful RWKV7_WKV  -> bit-identical (proves no regression).
  (B) FORWARD split-equivalence: fwd([A;B]) == [fwd(A); fwd(B, state0=final_A)]  -> exact.
  (C) BACKWARD truncated-BPTT: stateful grads (state carried+detached across the split) match a
      pure-PyTorch detached-carry reference (reference_rwkv7_stateful) within fp32 tolerance.

K is hardwired to 32 in the kernel. We use H=2, B=3, T=200, split at m=120.
"""
import torch
import rwkv.model  # loads RWKV_CUDA, registers torch.ops.rwkv
from rwkv.model.rwkv_ops import RWKV7_WKV, RWKV7_WKV_Stateful, reference_rwkv7_stateful

torch.manual_seed(0)
dev = "cuda"
B, T, H, K = 3, 200, 2, 32
M = 120  # split point


def make_inputs(dtype):
    """Realistic-ish WKV inputs. w in (0,1) (a decay), a in (0,1), k_deformed unit-normalized."""
    r = (0.5 * torch.randn(B, T, H, K, device=dev)).to(dtype)
    v = (0.5 * torch.randn(B, T, H, K, device=dev)).to(dtype)
    a = torch.sigmoid(torch.randn(B, T, H, K, device=dev)).to(dtype)
    kd = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev), dim=-1, p=2.0).to(dtype)
    k = (kd.float() * a.float()).to(dtype)  # matches rwkv_model: k = k_deformed * a
    w = (0.85 + 0.14 * torch.rand(B, T, H, K, device=dev)).float()  # fp32 required, near 1
    skip = torch.zeros(B, T, dtype=torch.bool, device=dev)
    # a few skips, never at t=0 of either chunk (kernel/prepare forbids skipping a chunk start)
    skip[:, 50] = True
    skip[:, 150] = True
    return r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), a.contiguous(), kd.contiguous(), skip.contiguous()


def maxabs(x, y):
    return (x.float() - y.float()).abs().max().item()


for dtype in [torch.float32, torch.bfloat16]:
    name = str(dtype).split(".")[-1]
    r, k, v, w, a, kd, skip = make_inputs(dtype)
    zeros_state = torch.zeros(B, H, K, K, dtype=torch.float32, device=dev)

    # ---- (A) stateful(state0=0) == non-stateful RWKV7_WKV ----
    out_ns = RWKV7_WKV.apply(r, k, v, w, a, kd, skip)
    out_s0, final_s0 = RWKV7_WKV_Stateful.apply(r, k, v, w, a, kd, skip, zeros_state)
    dA = maxabs(out_ns, out_s0)

    # ---- (B) forward split-equivalence ----
    def sl(x, lo, hi):
        return x[:, lo:hi].contiguous()

    rA, kA, vA, wA, aA, kdA, skA = (sl(t, 0, M) for t in (r, k, v, w, a, kd, skip))
    rB, kB, vB, wB, aB, kdB, skB = (sl(t, M, T) for t in (r, k, v, w, a, kd, skip))
    outA, finalA = RWKV7_WKV_Stateful.apply(rA, kA, vA, wA, aA, kdA, skA, zeros_state)
    outB, finalB = RWKV7_WKV_Stateful.apply(rB, kB, vB, wB, aB, kdB, skB, finalA.detach())
    out_split = torch.cat([outA, outB], dim=1)
    dB = maxabs(out_ns, out_split)

    # ---- (C) backward truncated-BPTT vs pure-PyTorch detached-carry reference ----
    grad_seed = torch.randn(B, T, H, K, device=dev).to(dtype)

    def leaf(x):
        y = x.clone().detach()
        y.requires_grad_(True)
        return y

    # CUDA stateful
    cA = [leaf(t) for t in (rA, kA, vA, aA, kdA)]
    cB = [leaf(t) for t in (rB, kB, vB, aB, kdB)]
    oA, fA = RWKV7_WKV_Stateful.apply(cA[0], cA[1], cA[2], wA, cA[3], cA[4], skA, zeros_state)
    oB, fB = RWKV7_WKV_Stateful.apply(cB[0], cB[1], cB[2], wB, cB[3], cB[4], skB, fA.detach())
    (torch.cat([oA, oB], dim=1).float() * grad_seed.float()).sum().backward()
    cuda_grads = [t.grad.float().clone() for t in cA] + [t.grad.float().clone() for t in cB]

    # pure-PyTorch reference (same detached carry => truncated BPTT)
    gA = [leaf(t) for t in (rA, kA, vA, aA, kdA)]
    gB = [leaf(t) for t in (rB, kB, vB, aB, kdB)]
    roA, rfA = reference_rwkv7_stateful(gA[0], gA[1], gA[2], wA, gA[3], gA[4], skA, None)
    roB, rfB = reference_rwkv7_stateful(gB[0], gB[1], gB[2], wB, gB[3], gB[4], skB, rfA.detach())
    (torch.cat([roA, roB], dim=1).float() * grad_seed.float()).sum().backward()
    ref_grads = [t.grad.float().clone() for t in gA] + [t.grad.float().clone() for t in gB]

    gnames = ["rA", "kA", "vA", "aA", "kdA", "rB", "kB", "vB", "aB", "kdB"]
    gdiffs = {n: maxabs(c, r_) for n, c, r_ in zip(gnames, cuda_grads, ref_grads)}
    dC = max(gdiffs.values())

    print(f"\n=== dtype={name} ===")
    print(f"  (A) stateful(0) vs non-stateful  max|d| = {dA:.3e}   (expect ~0)")
    print(f"  (B) forward split-equivalence    max|d| = {dB:.3e}   (expect ~0)")
    print(f"  (C) truncated-BPTT grad vs ref    max|d| = {dC:.3e}")
    for n, d in gdiffs.items():
        print(f"        grad {n:4s} max|d| = {d:.3e}")

print("\nDONE")
