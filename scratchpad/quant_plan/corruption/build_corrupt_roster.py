"""Stage 2 of the corruption roster: cap by section size and backfill to 50 users.

The top-50 by mean reviews/card contains 2.7M-3.9M-review users (5626, 6810, 6701, 6630) whose single
section is far past what one 11 GiB eval pass can hold, so they would OOM-skip and contribute no
evidence. They are recorded in `excluded` and the list is backfilled with the next-ranked users whose
section does fit.

Cap: MAX_ROWS rows in the user's section (the eval DB's own `length`, ~1.95x the review count). 200,000
rows ~= 100,000 reviews, which the 11 GiB cap handles; the biggest folded-in card histories are still
several hundred reviews, i.e. well past the 64/128 training cap this check exists to avoid.

Writes scratchpad/quant_plan/corruption/corrupt50.json
"""
import json
import sys
from pathlib import Path

import lmdb

DB = "F:/rwkv_lmdb/test_db_5k_adaptive92"
HERE = Path(__file__).resolve().parent
MAX_ROWS = 200_000
N = 50

doc = json.loads((HERE / "longhist50.json").read_text(encoding="utf-8"))
ranked = doc["ranked"]

env = lmdb.open(DB, map_size=250_000_000_000, readonly=True, lock=False)
lens = {}
with env.begin(write=False) as txn:
    for r in ranked:
        raw = txn.get(f"{r['user']}_batches".encode())
        if raw is None:
            lens[r["user"]] = None
            continue
        lens[r["user"]] = sum(b[2] for b in json.loads(raw))
env.close()

kept, excluded, missing = [], [], []
for r in ranked:
    if len(kept) >= N:
        break
    L = lens.get(r["user"])
    if L is None:
        missing.append(r["user"])
        continue
    if L > MAX_ROWS:
        excluded.append({**r, "section_rows": L})
        continue
    kept.append({**r, "section_rows": L})

out = {"pool": doc["pool"], "metric": doc["metric"], "max_section_rows": MAX_ROWS,
       "users": [r["user"] for r in kept], "selected": kept, "excluded_too_big": excluded,
       "missing_in_db": missing,
       "note": "ranked by mean reviews per card over users 5001-7500; users whose section exceeds "
               "MAX_ROWS are listed in excluded_too_big and replaced by the next-ranked user"}
(HERE / "corrupt50.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print(f"selected {len(kept)} users; section rows total {sum(r['section_rows'] for r in kept):,}")
print(f"excluded (section > {MAX_ROWS:,} rows): {[(r['user'], r['reviews'], r['section_rows']) for r in excluded]}")
print(f"missing in DB: {missing}")
print(f"\n{'user':>6} {'reviews':>9} {'cards':>7} {'mean':>6} {'maxcard':>8} {'rows':>9}")
for r in kept:
    print(f"{r['user']:>6} {r['reviews']:>9} {r['cards']:>7} {r['mean']:>6.1f} {r['max']:>8} "
          f"{r['section_rows']:>9,}")
print(f"\nlongest single-card history in the selected set: {max(r['max'] for r in kept)}")
