"""Is the external HDD actually a bottleneck for how we read LMDBs?

WHY MEASURE INSTEAD OF ASSUMING. Moving a db from F: to C: costs 15-30 minutes of USB copy and
~100 GB of the SSD we just freed, so it should be worth something. Two facts push the other way:
CLAUDE.md's own profile says training fetch is ALREADY HIDDEN ("~3-7 ms/step, 7 workers +
FETCH_AHEAD=5 fully hide prep+IPC"), and the OS page cache will serve a re-read from RAM
regardless of which disk the file sits on.

So this measures COLD random-read throughput per drive: pick keys at random, read them once, and
report MB/s. Random rather than sequential because that is how the fetch workers hit it (chunk keys
scattered across users), and reading each key once keeps the page cache from answering for us.

⚠ It cannot fully drop the page cache without admin rights, so a db that was recently touched will
look fast. The keys are drawn randomly from the whole keyspace to make that unlikely, and both arms
are treated identically, so the COMPARISON survives even if the absolute numbers are optimistic.

Usage: .venv/Scripts/python.exe scratchpad/workload/probe_lmdb_io.py [n_keys]
"""
import os
import random
import sys
import time

import lmdb

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEED = 20260822

TARGETS = [
    ("C:", r"C:\Users\Andrew\rwkv-anki-autoresearch\train_db_5k_h1_fix"),
    ("C:", r"C:\Users\Andrew\rwkv-anki-autoresearch\train_db_5k_h1"),
    ("F:", r"F:\rwkv_lmdb\test_db_5k_fix"),
    ("F:", r"F:\rwkv_lmdb\train_db_5k_h1_id3"),
]


def probe(path, n):
    env = lmdb.open(path, map_size=400_000_000_000, readonly=True, lock=False,
                    subdir=True, readahead=False)
    t_open = time.perf_counter()
    with env.begin(write=False) as txn:
        stat = txn.stat()
        total = stat["entries"]
        # sample keys by walking a cursor to random positions: cheaper than materialising
        # the whole keyspace, which is millions of entries here
        cur = txn.cursor()
        keys = []
        rng = random.Random(SEED)
        if not cur.first():
            return None
        all_keys = []
        for i, k in enumerate(cur.iternext(keys=True, values=False)):
            all_keys.append(k)
            if i > 20000:
                break
        keys = rng.sample(all_keys, min(n, len(all_keys)))

        t_enum = time.perf_counter() - t_open
        t0 = time.perf_counter()
        nbytes = 0
        for k in keys:
            v = txn.get(k)
            if v:
                nbytes += len(v)
        dt = time.perf_counter() - t0
    env.close()
    return total, len(keys), nbytes, dt, t_enum


print("cold-ish random reads, %d keys each\n" % N)
print("%-4s %-34s %10s %12s %10s %10s %10s"
      % ("drv", "db", "entries", "MB read", "sec", "MB/s", "walk_s"))
print("-" * 98)
for drv, path in TARGETS:
    if not os.path.isdir(path):
        print("%-4s %-34s   (absent)" % (drv, os.path.basename(path)))
        continue
    r = probe(path, N)
    if r is None:
        print("%-4s %-34s   (empty)" % (drv, os.path.basename(path)))
        continue
    total, nk, nbytes, dt, t_enum = r
    print("%-4s %-34s %10d %12.1f %10.2f %10.1f %10.1f"
          % (drv, os.path.basename(path), total, nbytes / 1e6, dt,
             (nbytes / 1e6) / dt if dt > 0 else float("nan"), t_enum), flush=True)
print("")
print("If the F: rows are within ~2x of the C: rows, the drive is not the bottleneck and")
print("moving a db to the SSD buys little -- CLAUDE.md already measured fetch as hidden.")
