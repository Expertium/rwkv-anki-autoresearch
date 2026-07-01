"""TEST: load the MODIFIED kernel from build/lib.../RWKV_CUDA.*.pyd (built with `build_ext` WITHOUT
--inplace, so the locked production copy is untouched) and (1) check it is BYTE-IDENTICAL to the
golden, and (2) time it back-to-back vs the golden's production bench. Does NOT import rwkv, so it
never loads the production .pyd -> no namespace collision."""
import glob
import sys
import time
from pathlib import Path
import importlib.util

import torch  # import torch first so its CUDA DLLs are on the loader path for the .pyd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kab_common import make_inputs, run_fwd_bwd

# Load the freshly-built (modified) extension by path.
cands = glob.glob(str(ROOT / "build" / "lib.*" / "rwkv" / "model" / "RWKV_CUDA*.pyd"))
assert cands, f"no build/ .pyd found under {ROOT/'build'} -- run the isolated build_ext first"
pyd = max(cands, key=lambda p: Path(p).stat().st_mtime)
print(f"loading MODIFIED kernel: {pyd}")
print(f"  mtime {time.ctime(Path(pyd).stat().st_mtime)}")
spec = importlib.util.spec_from_file_location("RWKV_CUDA", pyd)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # registers torch.ops.rwkv.*

GOLDEN = Path(__file__).resolve().parent / "golden.pt"


def main():
    golden = torch.load(GOLDEN)
    inp = make_inputs()
    out, ckpt, grads = run_fwd_bwd(*inp)
    torch.cuda.synchronize()

    got = {"out": out.cpu(), "ckpt": ckpt.cpu()}
    for i, gname in enumerate(["rg", "kg", "vg", "wg", "ag", "kdg"]):
        got[gname] = grads[i].cpu()

    print("\n=== PARITY (modified vs production golden; must be 0.0 -- allocator-only change) ===")
    worst = 0.0
    for name in ["out", "rg", "kg", "vg", "wg", "ag", "kdg"]:
        d = (got[name].float() - golden[name].float()).abs().max().item()
        worst = max(worst, d)
        print(f"  {name:5s} max|diff| = {d:.3e}")
    print(f"  WORST = {worst:.3e}  -> {'BIT-EXACT PASS' if worst == 0.0 else 'NONZERO (investigate)'}")

    N = 100
    for _ in range(3):
        run_fwd_bwd(*inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        run_fwd_bwd(*inp)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"\nMODIFIED_BENCH N={N} dt={dt:.4f}s  {N / dt:.2f} it/s  ({1e3 * dt / N:.3f} ms/iter)")
    print("(compare it/s vs PRODUCTION_BENCH from gen_golden.py -- directional only, contended box)")


if __name__ == "__main__":
    main()
