"""Size the roster users from the VAL LMDB (read-only) and print the distribution.

Used to decide the shard/solo settings of a roster eval: a user above the solo threshold must not
share the GPU with another big user (the 2026-07-12/13 wedge). Mirrors eval_sharded.user_sizes.
"""
import json
import sys
from pathlib import Path

import lmdb

VAL_DB = "F:/rwkv_lmdb/test_db_5k_adaptive92"
DB_SIZE = 250_000_000_000
ROOT = Path(__file__).resolve().parents[2]
roster = ROOT / "scratchpad" / "quant_plan" / "roster750_seed20261119.json"
users = json.loads(roster.read_text(encoding="utf-8"))["users"]

env = lmdb.open(VAL_DB, map_size=DB_SIZE, readonly=True, lock=False, max_readers=1024, readahead=False)
sizes = {}
with env.begin() as txn:
    for uid in users:
        raw = txn.get(f"{uid}_batches".encode())
        if raw is None:
            continue
        sizes[uid] = max(int(b[2]) for b in json.loads(raw))
env.close()

vals = sorted(sizes.values())
n = len(vals)
print(f"users sized: {n}  total work: {sum(vals):,}")
for q in (0.5, 0.9, 0.99, 0.999):
    print(f"  p{q*100:5.1f}: {vals[min(n - 1, int(q * n))]:,}")
print(f"  max  : {vals[-1]:,}")
for thr in (1_000_000, 500_000, 250_000):
    over = [u for u, s in sizes.items() if s >= thr]
    print(f"  users >= {thr:,}: {len(over)} {sorted(over)[:12]}")
print(f"first 12 roster users: {users[:12]}")
sys.exit(0)
