"""GOLDEN: run the CURRENT PRODUCTION kernel (rwkv/model/RWKV_CUDA.*.pyd, the pre-edit build) on
fixed-seed inputs and save outputs. Run this BEFORE deploying the Tier-1 change in-place. Loading
the production .pyd here is a shared-read open -- it does NOT conflict with the export workers that
also have it loaded. Also times the production kernel back-to-back (directional speed reference)."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rwkv.model  # noqa: F401  -> triggers `from . import RWKV_CUDA` (production kernel registration)
from kab_common import make_inputs, run_fwd_bwd

OUT = Path(__file__).resolve().parent / "golden.pt"


def main():
    inp = make_inputs()
    out, ckpt, grads = run_fwd_bwd(*inp)
    torch.cuda.synchronize()
    payload = {"out": out.cpu(), "ckpt": ckpt.cpu()}
    for i, gname in enumerate(["rg", "kg", "vg", "wg", "ag", "kdg"]):
        payload[gname] = grads[i].cpu()
    torch.save(payload, OUT)
    print(f"GOLDEN saved -> {OUT}")
    print(f"  out {tuple(out.shape)}  ckpt {tuple(ckpt.shape)}  |out| max {out.abs().max().item():.4f}")

    # Directional microbench: many fwd+bwd back-to-back, ONE sync. The production path does a
    # synchronizing cudaFree inside every fwd and every bwd, so 2*N forced syncs here.
    N = 100
    for _ in range(3):  # warmup
        run_fwd_bwd(*inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        run_fwd_bwd(*inp)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"PRODUCTION_BENCH N={N} dt={dt:.4f}s  {N / dt:.2f} it/s  ({1e3 * dt / N:.3f} ms/iter)")


if __name__ == "__main__":
    main()
