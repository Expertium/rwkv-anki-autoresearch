"""Paired simultaneous throughput trials + one-sided Wilcoxon (protocol point 7-8).

One trial launches `--threads` BEFORE bench processes and `--threads` AFTER bench processes
ALL AT THE SAME TIME (each runs the Rust engine for `--secs` on the same user's trace), so
external load hits both sides equally. Sum each side's reviews -> one paired point
(before_total, after_total). Repeat `--trials` times (drop `--warmup`). Accept the speedup
iff one-sided Wilcoxon signed-rank p < 0.01 (H1: after faster).

Usage:
  python optimization/wilcoxon_speed.py --before reference/rwkv_ref_558.safetensors \
      --after reference/rwkv_iter3_62.safetensors [--secs 20 --threads 3 --trials 10 --warmup 1]
Prints median throughputs, speedup, and WILCOXON_P <p>.
"""
import argparse
import os
import re
import statistics
import subprocess
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
BIN = str(ROOT / "rust" / "rwkv-infer" / "target" / "release" / "rwkv-infer.exe")


def launch(weights, secs, user):
    env = {**os.environ, "RWKV_WEIGHTS": weights, "OMP_NUM_THREADS": "1"}
    return subprocess.Popen(
        [BIN, "--bench", str(secs), str(user)], cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )


def reviews(proc):
    out, _ = proc.communicate()
    m = re.search(r"BENCH reviews=(\d+)", out)
    if not m:
        raise SystemExit(f"no BENCH line:\n{out}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--user", type=int, default=107)
    a = ap.parse_args()

    pairs = []
    for trial in range(a.trials + a.warmup):
        # launch all before+after procs simultaneously
        bps = [launch(a.before, a.secs, a.user) for _ in range(a.threads)]
        aps = [launch(a.after, a.secs, a.user) for _ in range(a.threads)]
        b = sum(reviews(p) for p in bps)
        aa = sum(reviews(p) for p in aps)
        tag = "warmup" if trial < a.warmup else "trial "
        print(f"{tag} {trial}: before={b} after={aa} ratio={aa/b:.3f}", flush=True)
        if trial >= a.warmup:
            pairs.append((b, aa))

    bef = [p[0] for p in pairs]
    aft = [p[1] for p in pairs]
    # to reviews/sec/thread for readability
    med_b = statistics.median(bef) / a.threads / a.secs
    med_a = statistics.median(aft) / a.threads / a.secs
    diffs = [aa - b for b, aa in pairs]
    try:
        stat, p = wilcoxon(diffs, alternative="greater")
    except ValueError as e:  # e.g. all-zero diffs
        p = 1.0
        print("wilcoxon note:", e)
    print(f"\nmedian throughput: before {med_b:.1f} rev/s/thread, after {med_a:.1f} "
          f"(speedup {med_a/med_b:.3f}x over {len(pairs)} trials, {a.threads} threads)")
    print(f"WILCOXON_P {p:.3e}")
    print("PASS (p<0.01)" if p < 0.01 else "NOT SIGNIFICANT (p>=0.01)")


if __name__ == "__main__":
    main()
