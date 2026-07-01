"""Attribute WKV time: forward-only vs backward-only (using the build/ modified .pyd). Grounds the
Tier-3 (chunked tensor-core kernel) decision -- the rewrite's payoff is concentrated wherever the
time is. Backward recomputes the forward from checkpoints + does the gradient ops, so it is expected
to dominate. Loads the build/ .pyd by path (no rebuild needed)."""
import glob
import time
from pathlib import Path
import importlib.util

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kab_common import make_inputs

pyd = max(glob.glob(str(ROOT / "build" / "lib.*" / "rwkv" / "model" / "RWKV_CUDA*.pyd")),
          key=lambda p: Path(p).stat().st_mtime)
spec = importlib.util.spec_from_file_location("RWKV_CUDA", pyd)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def bench(fn, N=200, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1e3  # ms/call


def main():
    r, k, v, w, a, kd, skip, grad = make_inputs()
    fwd = lambda: torch.ops.rwkv.rwkv7_wkv_forward_float.default(r, k, v, w, a, kd, skip)
    out, ckpt = fwd()
    torch.cuda.synchronize()
    bwd = lambda: torch.ops.rwkv.rwkv7_wkv_backward_float.default(r, k, v, w, a, kd, skip, ckpt, grad)

    f_ms = bench(fwd)
    b_ms = bench(bwd)
    tot = f_ms + b_ms
    print(f"shape B{r.shape[0]} T{r.shape[1]} H{r.shape[2]} K{r.shape[3]}")
    print(f"  forward   {f_ms:.4f} ms/call  ({100*f_ms/tot:.1f}% of fwd+bwd)")
    print(f"  backward  {b_ms:.4f} ms/call  ({100*b_ms/tot:.1f}% of fwd+bwd)")
    print(f"  bwd/fwd ratio = {b_ms/f_ms:.2f}x")


if __name__ == "__main__":
    main()
