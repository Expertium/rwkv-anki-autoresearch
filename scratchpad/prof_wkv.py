"""Per-kernel WKV profile to bound the tensor-core upside.

The ONLY matrix-matrix products in the whole WKV op are in the scan (parallel_scan.cu:
rwkv7_scan_kernel + rwkv7_add_kernel) -- everything else (base/final/sequential fwd+bwd) is
matrix-VECTOR done via warp-shuffle reductions, which tensor cores can't accelerate. So the
low-risk 'tensor-core the scan' win is CAPPED at the scan kernels' share of GPU time. This script
measures that share at a few realistic champion-regime shapes (H=2, K=16, long T -> time-parallel
path). Loads the PRODUCTION (deployed Tier-1) .pyd by path so no rwkv import chain is needed.
"""
import glob
import importlib.util
import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "kernel_ab"))  # kab_common
from kab_common import make_inputs

# Load the deployed production kernel (registers torch.ops.rwkv.*)
pyd = glob.glob(str(ROOT / "rwkv" / "model" / "RWKV_CUDA*.pyd"))[0]
print(f"loaded kernel: {pyd}")
spec = importlib.util.spec_from_file_location("RWKV_CUDA", pyd)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def bucket(name):
    n = name.lower()
    if "scan_kernel" in n or "add_kernel" in n:
        return "scan (MATMUL -> tensor-core-able)"
    if "base_kernel" in n or "final_kernel" in n or "wkv_forward_kernel" in n or "wkv_backward_kernel" in n:
        return "recurrence (matrix-vector, warp-shuffle)"
    return "other (memset/copy/elementwise)"


def prof_shape(B, T, H, K, N=100, warm=15):
    r, k, v, w, a, kd, skip, grad = make_inputs(B=B, T=T, H=H, K=K)

    def step():
        out, ckpt = torch.ops.rwkv.rwkv7_wkv_forward_float.default(r, k, v, w, a, kd, skip)
        torch.ops.rwkv.rwkv7_wkv_backward_float.default(r, k, v, w, a, kd, skip, ckpt, grad)

    for _ in range(warm):
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(N):
            step()
        torch.cuda.synchronize()

    def cuda_us(e):  # self GPU time in microseconds, across torch versions
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                return v
        return 0.0

    tot = {}
    per_kernel = {}
    for e in prof.key_averages():
        t = cuda_us(e)  # microseconds
        if t <= 0:
            continue
        b = bucket(e.key)
        tot[b] = tot.get(b, 0.0) + t
        per_kernel[e.key] = per_kernel.get(e.key, 0.0) + t

    grand = sum(tot.values())
    print(f"\n===== B{B} T{T} H{H} K{K}  ({N} fwd+bwd iters) =====")
    print(f"total GPU kernel time: {grand/1e3:.2f} ms  ({grand/1e3/N:.4f} ms/iter)")
    for b in sorted(tot, key=lambda x: -tot[x]):
        print(f"  {tot[b]/grand*100:6.2f}%  {b}")
    print("  --- top kernels ---")
    for name in sorted(per_kernel, key=lambda x: -per_kernel[x])[:8]:
        short = name if len(name) < 70 else name[:67] + "..."
        print(f"    {per_kernel[name]/grand*100:6.2f}%  {short}")
    scan_share = tot.get("scan (MATMUL -> tensor-core-able)", 0.0) / grand * 100
    print(f"  >>> TENSOR-CORE CEILING (scan share) = {scan_share:.2f}%  "
          f"of WKV GPU time <<<")
    return scan_share


if __name__ == "__main__":
    shares = {}
    for shape in [(8, 1600, 2, 16), (8, 8000, 2, 16), (8, 16000, 2, 16), (16, 30000, 2, 16)]:
        shares[shape] = prof_shape(*shape)
    print("\n================ SUMMARY: scan (matmul) share by shape ================")
    for shape, s in shares.items():
        print(f"  B{shape[0]} T{shape[1]}: {s:.2f}%")
