"""Throughput vs batch size, swept across RAYON thread counts.

The 32-thread Pareto sweep back-bent past B=128 -- this checks whether that's rayon oversubscription
on tiny (K=32) matmuls. One `--bench-synth` subprocess per (threads, B); plot one line per thread count.

Usage: .venv/Scripts/python.exe scratchpad/thread_sweep.py [secs_per_point]
Outputs: scratchpad/thread_sweep.csv, scratchpad/thread_sweep.png
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "rust" / "rwkv-infer" / "target" / "release" / "rwkv-infer.exe"
WEIGHTS = "reference/rwkv_iter36_124.safetensors"
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

THREADS = [1, 2, 4, 8, 16, 32]
BATCHES = [16, 64, 128, 256, 512, 1024, 2048]
REV_RE = re.compile(r"rev_s\s+([0-9.]+)")


def bench(threads: int, b: int) -> float:
    env = dict(os.environ)
    env["RWKV_WEIGHTS"] = WEIGHTS
    env["RAYON_NUM_THREADS"] = str(threads)
    out = subprocess.run(
        [str(BIN), "--bench-synth", str(SECS), str(b)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=str(ROOT), text=True,
    ).stdout
    m = REV_RE.search(out or "")
    return float(m.group(1)) if m else float("nan")


def main():
    grid = {}  # threads -> {B: rev_s}
    print(f"{'threads':>8} " + " ".join(f"{b:>8}" for b in BATCHES))
    for t in THREADS:
        row = {}
        for b in BATCHES:
            row[b] = bench(t, b)
        grid[t] = row
        print(f"{t:>8} " + " ".join(f"{row[b]:>8.0f}" for b in BATCHES))

    csv = ROOT / "scratchpad" / "thread_sweep.csv"
    with open(csv, "w") as f:
        f.write("threads," + ",".join(str(b) for b in BATCHES) + "\n")
        for t in THREADS:
            f.write(f"{t}," + ",".join(f"{grid[t][b]:.0f}" for b in BATCHES) + "\n")
    print(f"wrote {csv}")

    # best (threads,B) overall
    best = max(((grid[t][b], t, b) for t in THREADS for b in BATCHES))
    print(f"BEST: {best[0]:.0f} rev/s at threads={best[1]} B={best[2]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; CSV only")
        return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for t in THREADS:
        ax.plot(BATCHES, [grid[t][b] for b in BATCHES], "o-", label=f"{t} thread(s)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Throughput (reviews/s)")
    ax.set_title("Batched query throughput vs B, per thread count (iter36, CPU)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = ROOT / "scratchpad" / "thread_sweep.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
