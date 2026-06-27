"""Measure single-stream (B=1) Rust throughput for a model checkpoint.

Exports the .pth weights to safetensors, runs the Rust engine on the 3 reference users
(reusing the model-independent trace features), and prints the median rev/s. Same method
as the iteration-0 baseline (181.8 rev/s), so the numbers are comparable across iterations.

IMPORTANT: the Rust engine's dims (rust/rwkv-infer/src/model.rs H/C/STREAM_LAYERS) must
match the model's architecture, and the release binary must be rebuilt after any dim change.

Usage: python optimization/measure_throughput.py pretrain/rwkv/opt_iter4/rwkv_iter4_62.pth
Prints: THROUGHPUT <median rev/s>
"""
import re
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "rust" / "rwkv-infer" / "target" / "release" / "rwkv-infer.exe"
USERS = ["107", "136", "156"]


def main():
    pth = sys.argv[1]
    out = ROOT / "reference" / "_bench_weights.safetensors"
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    save_file({k: v.detach().cpu().contiguous().float() for k, v in sd.items()}, str(out))

    env = {"RWKV_WEIGHTS": str(out), "OMP_NUM_THREADS": "7"}
    import os
    full_env = {**os.environ, **env}
    res = subprocess.run(
        [str(BIN), *USERS], cwd=str(ROOT), env=full_env,
        capture_output=True, text=True,
    )
    rates = [float(m) for m in re.findall(r"\(([\d.]+) rev/s\)", res.stdout)]
    if not rates:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit("no rev/s parsed — is the binary built and dims correct?")
    med = statistics.median(rates)
    print(f"per-user rev/s: {rates}")
    print(f"THROUGHPUT {med:.1f}")
    return med


if __name__ == "__main__":
    main()
