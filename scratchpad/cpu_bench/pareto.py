"""Speed-vs-RAM Pareto frontier of the batched RWKV query forward (CPU-speed phase).

For each B in {1,2,...,maxB} spawn ONE `rwkv-infer --bench-synth <secs> <B>` subprocess (synthetic
warmed states of the right shapes -> no warmup, identical compute+memory), poll peak RSS via psutil,
parse throughput. SINGLE-THREAD by default (RAYON=OMP=1) to match the current 1-thread phase.

Env: RWKV_BIN (default champion snapshot), RWKV_WEIGHTS (default champ_h2k16),
     RWKV_PARETO_THREADS (default 1).
Usage: python scratchpad/cpu_bench/pareto.py [secs_per_B] [maxB] [out_tag]
Writes scratchpad/cpu_bench/pareto_<tag>.csv (+ .png if matplotlib).
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
BIN = os.environ.get("RWKV_BIN", str(ROOT / "scratchpad" / "cpu_bench" / "champion.exe"))
WEIGHTS = os.environ.get("RWKV_WEIGHTS", "reference/champ_h2k16.safetensors")
THREADS = os.environ.get("RWKV_PARETO_THREADS", "1")

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
MAXB = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
TAG = sys.argv[3] if len(sys.argv) > 3 else "h2k16"

REV_RE = re.compile(r"rev_s\s+([0-9.]+)")


def bench_one(b: int):
    env = dict(os.environ)
    env["RWKV_WEIGHTS"] = WEIGHTS
    env["OMP_NUM_THREADS"] = THREADS
    env["RAYON_NUM_THREADS"] = THREADS
    p = subprocess.Popen([BIN, "--bench-synth", str(SECS), str(b)],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
                         cwd=str(ROOT), text=True)
    proc = psutil.Process(p.pid)
    peak = 0
    while p.poll() is None:
        try:
            peak = max(peak, proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(0.01)
    out = p.stdout.read()
    p.wait()
    m = REV_RE.search(out or "")
    return (float(m.group(1)) if m else float("nan")), peak


def main():
    bs = []
    b = 1
    while b <= MAXB:
        bs.append(b)
        b *= 2
    print(f"# bin={Path(BIN).name} weights={WEIGHTS} threads={THREADS} secs/B={SECS}")
    print(f"{'B':>6} {'rev_s':>10} {'peak_MB':>10} {'rev_s/MB':>10}")
    rows = []
    for b in bs:
        rev_s, peak = bench_one(b)
        mb = peak / 1e6
        eff = rev_s / mb if mb else 0.0
        rows.append((b, rev_s, mb, eff))
        print(f"{b:>6} {rev_s:>10.1f} {mb:>10.1f} {eff:>10.2f}", flush=True)
    csv = ROOT / "scratchpad" / "cpu_bench" / f"pareto_{TAG}.csv"
    with open(csv, "w") as f:
        f.write("B,rev_s,peak_MB,rev_s_per_MB\n")
        for b, rev_s, mb, eff in rows:
            f.write(f"{b},{rev_s:.1f},{mb:.1f},{eff:.3f}\n")
    print(f"wrote {csv}")
    # peak throughput + best efficiency
    bestT = max(rows, key=lambda r: r[1])
    bestE = max(rows, key=lambda r: r[3])
    print(f"PEAK THROUGHPUT: B={bestT[0]}  {bestT[1]:.0f} rev/s  ({bestT[2]:.1f} MB)")
    print(f"BEST rev/s/MB : B={bestE[0]}  {bestE[3]:.1f} rev/s/MB  ({bestE[1]:.0f} rev/s, {bestE[2]:.1f} MB)")


if __name__ == "__main__":
    main()
