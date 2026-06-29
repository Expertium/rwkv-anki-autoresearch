"""Speed-vs-RAM Pareto frontier of the batched RWKV query forward.

For each batch size B in {1,2,4,...,maxB} we spawn ONE `rwkv-infer --bench-synth <secs> <B>`
subprocess (synthetic warmed states of the right shapes -> no warmup, identical compute+memory),
poll its peak RSS via psutil while it runs, and parse its throughput. Then plot:
  (1) throughput vs batch size (log x), and
  (2) the Pareto frontier: throughput (rev/s) vs peak RAM (MB), each point labelled with B.

Usage: .venv/Scripts/python.exe scratchpad/sweep_pareto.py [secs_per_B] [maxB]
Outputs: scratchpad/pareto_data.csv, scratchpad/throughput_vs_batch.png, scratchpad/pareto_speed_ram.png
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "rust" / "rwkv-infer" / "target" / "release" / "rwkv-infer.exe"
WEIGHTS = "reference/rwkv_iter36_124.safetensors"

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
MAXB = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

REV_RE = re.compile(r"rev_s\s+([0-9.]+)")


def bench_one(b: int):
    """Spawn the synth bench at batch b; return (rev_s, peak_rss_bytes)."""
    env = dict(os.environ)
    env["RWKV_WEIGHTS"] = WEIGHTS
    p = subprocess.Popen(
        [str(BIN), "--bench-synth", str(SECS), str(b)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=str(ROOT),
        text=True,
    )
    proc = psutil.Process(p.pid)
    peak = 0
    while p.poll() is None:
        try:
            peak = max(peak, proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(0.015)
    out = p.stdout.read()
    p.wait()
    m = REV_RE.search(out or "")
    rev_s = float(m.group(1)) if m else float("nan")
    return rev_s, peak


def main():
    bs = []
    b = 1
    while b <= MAXB:
        bs.append(b)
        b *= 2
    print(f"# threads(all cores)={psutil.cpu_count()}  secs/B={SECS}  weights={WEIGHTS}")
    print(f"{'B':>6} {'rev_s':>10} {'peak_MB':>10} {'rev_s/MB':>10}")
    rows = []
    for b in bs:
        rev_s, peak = bench_one(b)
        mb = peak / 1e6
        eff = rev_s / mb if mb else 0.0
        rows.append((b, rev_s, mb, eff))
        print(f"{b:>6} {rev_s:>10.1f} {mb:>10.1f} {eff:>10.2f}")

    csv = ROOT / "scratchpad" / "pareto_data.csv"
    with open(csv, "w") as f:
        f.write("B,rev_s,peak_MB,rev_s_per_MB\n")
        for b, rev_s, mb, eff in rows:
            f.write(f"{b},{rev_s:.1f},{mb:.1f},{eff:.3f}\n")
    print(f"wrote {csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV written, skipping plots")
        return

    B = [r[0] for r in rows]
    rev = [r[1] for r in rows]
    mb = [r[2] for r in rows]

    # (1) throughput vs batch size
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(B, rev, "o-", color="#1f77b4")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Throughput (reviews/s)")
    ax.set_title("Batched query throughput vs batch size (iter36, CPU)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = ROOT / "scratchpad" / "throughput_vs_batch.png"
    fig.savefig(p1, dpi=130)
    print(f"wrote {p1}")

    # (2) Pareto frontier: throughput vs RAM, labelled by B
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(mb, rev, "o-", color="#d62728")
    for b, x, y in zip(B, mb, rev):
        ax.annotate(f"B={b}", (x, y), textcoords="offset points", xytext=(6, -2), fontsize=8)
    ax.set_xlabel("Peak RAM (MB)")
    ax.set_ylabel("Throughput (reviews/s)")
    ax.set_title("Speed vs RAM Pareto frontier (batched query, iter36)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = ROOT / "scratchpad" / "pareto_speed_ram.png"
    fig.savefig(p2, dpi=130)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
