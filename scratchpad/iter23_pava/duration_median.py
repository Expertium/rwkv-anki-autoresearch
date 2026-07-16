"""Compute the GLOBAL train-set median review duration (users 1-5000) for the iter-23
probe-row imputation constant (MONOTONICITY_PLAN.md stage 2, Andrew delegated 2026-07-16).

Durations are integer milliseconds (Anki caps answer time), so an exact median comes from
an integer histogram accumulated across users -- no global sort needed. The deploy/train
constant is the RAW-ms median; the probe row carries scale_duration(median) in feature
col 8 (median is transform-invariant under the monotone log-z map).

Reads only (dataset is read-only). ~5000 single-column parquet scans, pyarrow, 7 threads.
"""

import json
import os
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(7)

DATA = r"C:\Users\Andrew\anki-revlogs-10k\revlogs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "duration_median.json")

counts = Counter()
total = 0
t0 = time.time()
for uid in range(1, 5001):
    path = os.path.join(DATA, f"user_id={uid}", "data.parquet")
    if not os.path.exists(path):
        print(f"MISSING user {uid}")
        continue
    tbl = pq.read_table(path, columns=["duration"])
    vc = tbl.column("duration").value_counts()
    for item in vc:
        counts[item["values"].as_py()] += item["counts"].as_py()
    total += tbl.num_rows
    if uid % 500 == 0:
        print(f"user {uid}: {total:,} rows, {len(counts)} distinct, "
              f"{time.time() - t0:.0f}s", flush=True)

# exact median from the histogram (lower median for even totals)
target = (total + 1) // 2
acc = 0
median = None
for v in sorted(counts):
    acc += counts[v]
    if acc >= target:
        median = v
        break

import numpy as np
scaled = float((np.log(10 + median) - 8.9) / 1.07)  # scale_duration, data_processing.py
result = {
    "train_users": "1-5000",
    "n_reviews": total,
    "median_duration_ms": int(median),
    "scaled_duration_constant": scaled,
    "distinct_values": len(counts),
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
