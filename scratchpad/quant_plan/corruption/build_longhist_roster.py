"""Build the long-history roster for the corruption check (Andrew, 2026-11-21 22:20).

"For each user, calculate the average card history length... find 50 users with longest average review
histories."  The pool is the eval set users 5001-7500 (the same 2,500 users the 750-user roster is drawn
from), so the check also covers users that the random 750 roster happens to miss.

Reads only the `card_id` column of each user's raw revlogs parquet (the eval DB only stores the derived
sections), so the scan is I/O-bound and fast.

Writes scratchpad/quant_plan/corruption/longhist50.json:
  {"pool": [5001, 7500], "metric": "mean reviews per card", "ranked": [...], "users": [...50...]}
`ranked` keeps the top 300 with total reviews, cards, mean and max, for the record.
"""
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path("C:/Users/Andrew/anki-revlogs-10k-id/revlogs")
OUT = Path(__file__).resolve().parent / "longhist50.json"
POOL = (5001, 7500)
TOP_N = 50
KEEP_RANKED = 300


def one(user):
    p = RAW / f"user_id={user}" / "data.parquet"
    if not p.exists():
        return None
    cards = pq.read_table(p, columns=["card_id"]).column("card_id").to_pylist()
    if not cards:
        return None
    cnt = Counter(cards)
    n = len(cards)
    nc = len(cnt)
    return {"user": user, "reviews": n, "cards": nc, "mean": n / nc, "max": max(cnt.values())}


def main():
    users = list(range(POOL[0], POOL[1] + 1))
    with mp.Pool(processes=8) as pool:
        rows = [r for r in pool.imap_unordered(one, users, chunksize=8) if r]
    rows.sort(key=lambda r: (-r["mean"], -r["max"], r["user"]))
    ranked = rows[:KEEP_RANKED]
    top = rows[:TOP_N]
    out = {"pool": list(POOL), "metric": "mean reviews per card", "n_users_scanned": len(rows),
           "ranked": ranked, "users": [r["user"] for r in top], "top": top}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"scanned {len(rows)} users; top {TOP_N} by mean reviews/card:")
    print(f"{'rank':>4} {'user':>6} {'reviews':>8} {'cards':>7} {'mean':>7} {'max':>5}")
    for i, r in enumerate(top, 1):
        print(f"{i:>4} {r['user']:>6} {r['reviews']:>8} {r['cards']:>7} {r['mean']:>7.1f} {r['max']:>5}")
    tot = sum(r["reviews"] for r in top)
    print(f"\ntotal reviews over the {TOP_N} users: {tot:,}  "
          f"(min {min(r['reviews'] for r in top):,}, max {max(r['reviews'] for r in top):,})")
    print(f"total cards: {sum(r['cards'] for r in top):,}; "
          f"longest single card history: {max(r['max'] for r in top)}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
