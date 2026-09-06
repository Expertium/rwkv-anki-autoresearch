"""How much of the SCORED metric is exposed to a cold chunk start?

The learned-initial-state lever (round 2026-09-06 rank 2) can only pay where the eval actually
restarts a stream cold. The eval chunks each user the same way training does, so:
  - a user whose whole history fits in ONE chunk pays the cold start only at review 0, and the
    first sixth of a history is never scored anyway (TimeSeriesSplit(5)) -> ~no exposure;
  - a user with N chunks has N-1 interior boundaries, each followed by a recovery window.
This counts, over the VAL eval db, the number of chunks per user and the share of SCORED rows that
fall within W rows after an INTERIOR boundary -- the rows a learned init could help.
Then it projects the by-user mean ahead gain as (exposed share) x (measured reset cost in that
window), using the two measured recovery curves (users 102 / 106).
Read-only on the LMDB batch lists (no model, no GPU). Usage: chunk_exposure.py [W=2048]
"""
import json
import os
import sys

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
import lmdb
import numpy as np

W = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
DB = "F:/rwkv_lmdb/test_db_5k_id5"
env = lmdb.open(DB, map_size=400_000_000_000, readonly=True, lock=False)
n_chunks, rows_tot, exposed_rows = [], [], []
users = []
with env.begin(write=False) as txn:
    for uid in range(5001, 7501):
        if uid == 6701:
            continue
        raw = txn.get(f"{uid}_batches".encode())
        if raw is None:
            continue
        bs = json.loads(raw)            # each entry: [start_th, end_th, n_rows]
        users.append(uid)
        n_chunks.append(len(bs))
        tot = sum(b[2] for b in bs)
        rows_tot.append(tot)
        # rows within W of an INTERIOR boundary = the first W rows of every chunk after the first
        exposed = sum(min(W, b[2]) for b in bs[1:])
        exposed_rows.append(exposed)
env.close()
n_chunks = np.array(n_chunks); rows_tot = np.array(rows_tot); exposed_rows = np.array(exposed_rows)
print(f"VAL users read: {len(users)}  (db {DB}, W={W})")
print(f"chunks per user: 1 -> {(n_chunks == 1).sum()} users ({(n_chunks == 1).mean():.1%}), "
      f"2 -> {(n_chunks == 2).sum()}, 3+ -> {(n_chunks >= 3).sum()}, max {n_chunks.max()}")
print(f"rows per user: median {np.median(rows_tot):,.0f}  mean {rows_tot.mean():,.0f}  max {rows_tot.max():,}")
frac = exposed_rows / np.maximum(rows_tot, 1)
print(f"share of a user's rows within {W} of an interior boundary: by-user mean {frac.mean():.4f}, "
      f"median {np.median(frac):.4f}, p90 {np.percentile(frac, 90):.4f}")
print(f"users with ANY interior boundary: {(n_chunks > 1).sum()} ({(n_chunks > 1).mean():.1%})")
# projection: the measured reset cost in the first W rows, from reset_cost_screen (users 102 / 106)
for label, cost in (("user 102 recovery curve", 0.0043), ("user 106 recovery curve", 0.0118)):
    gain = frac.mean() * cost
    print(f"projected by-user mean ahead gain with a PERFECT init ({label}, mean cost over the first "
          f"{W} rows ~{cost:+.4f}): {gain:+.6f}")
print("\nNOTE: this is an UPPER BOUND -- it assumes a learned init recovers the whole cold-start cost,")
print("and it ignores that the metric scores only label_is_equalize rows (which skew LATE in a")
print("history, i.e. away from a user's first chunk but not from later boundaries).")
