#!/usr/bin/env python
"""How much of the rebuild's cost is FIXED startup vs per-review work? CPU, ~4 min.

FUTURE_FEATURES.md's ~23 h projection comes from ONE point: a 20-user probe at 6,671 reviews/s.
The 100-user de-risk build just did 5,226,500 reviews in about a minute per arm, i.e. roughly 10x
that rate -- and its users are slightly LARGER than the probe's (52,265 vs 45,697 reviews each), so
"the sample was easy" does not explain it.

The likely explanation is that a single-point rate cannot separate the two costs. Worker spawn,
imports, the LMDB open and the map_size reservation are paid ONCE; at 20 users they amortize over
5x fewer reviews than at 100. If that is what happened, the probe measured mostly startup and the
23 h figure is a large overestimate -- which matters for scheduling, since a few hours is a
"do it now" job and a day is not.

TWO POINTS SOLVE IT: t = fixed + reviews/rate. Run 20 and 100 users back to back, same process
pool, same disk, same contention, and fit both terms instead of quoting a ratio.

ASCII output only.
"""
import io
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOML = """DEVICE = "cpu"
DTYPE = "bfloat16"
DATA_PATH = "../anki-revlogs-10k-id"
LMDB_PATH = "F:/rwkv_lmdb/timing_db"
LMDB_SIZE = 8_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 40_000_000_000
USER_START = 1
USER_END = {end}
MAX_BATCH_SIZE = 16384
PROCESSES = 6
RWKV_SUBMODULES = ["card_id", "note_id", "deck_id", "preset_id", "user_id"]
"""


def reviews(lo, hi):
    import pandas as pd
    from pathlib import Path
    d = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
    n = 0
    for u in range(lo, hi + 1):
        p = d / "revlogs" / f"user_id={u}"
        if p.is_dir():
            n += len(pd.read_parquet(p, columns=["card_id"]))
    return n


def run(end):
    cfg = os.path.join(REPO, "scratchpad", "id_features", "dp_timing.toml")
    io.open(cfg, "w", encoding="utf-8", newline="\n").write(TOML.format(end=end))
    shutil.rmtree("F:/rwkv_lmdb/timing_db", ignore_errors=True)
    env = dict(os.environ, RWKV_ID_FEATURES="1", OMP_NUM_THREADS="2")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "rwkv.data_processing", "--config",
                        "scratchpad/id_features/dp_timing.toml"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    assert p.returncode == 0, (p.stdout[-800:] + p.stderr[-800:])
    return dt


def main():
    pts = []
    for end in (20, 100):
        n = reviews(1, end)
        dt = run(end)
        pts.append((n, dt))
        print(f"  {end:3d} users, {n:>10,} reviews: {dt:7.1f} s  ({n/dt:9,.0f} rev/s naive)")
    (n1, t1), (n2, t2) = pts
    rate = (n2 - n1) / (t2 - t1)
    fixed = t1 - n1 / rate
    print(f"\n  FIT: fixed = {fixed:.1f} s, marginal = {rate:,.0f} reviews/s")
    print(f"  (the single-point 'rate' at 20 users was {n1/t1:,.0f} rev/s -- "
          f"{100*fixed/t1:.0f}% of that run was startup)")
    for label, n in (("train half 1-5000", 372_000_000), ("test half", 186_000_000)):
        print(f"  projected {label:20s} ({n/1e6:.0f} M reviews): {(fixed + n/rate)/3600:6.2f} h")
    print(f"  projected BOTH: {(2*fixed + 558_000_000/rate)/3600:.2f} h "
          f"(FUTURE_FEATURES.md currently says ~23 h)")
    shutil.rmtree("F:/rwkv_lmdb/timing_db", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
