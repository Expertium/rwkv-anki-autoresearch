"""Probe: section (batch) lengths in the eval DB + per-user card-history statistics.

Answers two questions before the corruption check is designed:
 1. does the eval DB section a user's history, and how long is a section?  (truncation evidence)
 2. for users 5001-7500, what is the mean and max number of reviews per card?  (roster selection)
"""
import json
import sys
from pathlib import Path

import lmdb

ROOT = Path(__file__).resolve().parents[2]
DB = "F:/rwkv_lmdb/test_db_5k_adaptive92"
RAW = Path("C:/Users/Andrew/anki-revlogs-10k-id/revlogs")

users = [int(x) for x in sys.argv[1:]] or [5001, 5002, 5003]

env = lmdb.open(DB, map_size=250_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    stat = txn.stat()
    print("db entries:", stat["entries"])
    for u in users:
        raw = txn.get(f"{u}_batches".encode())
        if raw is None:
            print(f"user {u}: NO BATCHES in {DB}")
            continue
        batches = json.loads(raw)
        lens = [b[2] for b in batches]
        print(f"user {u}: {len(batches)} section(s), lengths={lens}, total={sum(lens)}, "
              f"spans={[ (b[0], b[1]) for b in batches ][:4]}")
env.close()

for u in users:
    p = RAW / f"user_id={u}" / "data.parquet"
    if not p.exists():
        print(f"user {u}: no raw revlogs")
        continue
    import pyarrow.parquet as pq
    col = pq.read_table(p, columns=["card_id"]).column("card_id").to_pylist()
    from collections import Counter
    cnt = Counter(col)
    n = len(col)
    ncards = len(cnt)
    mx = max(cnt.values())
    print(f"user {u}: reviews={n} cards={ncards} mean_reviews_per_card={n/ncards:.1f} "
          f"max_reviews_per_card={mx}")
