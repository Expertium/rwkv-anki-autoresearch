"""Why featB's fetch worker died with KeyError in insert_probes.

THE CRASH. prepare_batch.insert_probes:123 builds
    q_map = {review_th -> query row}   over rows with is_query
    query_old = [q_map[review_th[r]] for r in pick]
and died on KeyError, i.e. a PICKED TARGET ROW HAS NO QUERY ROW. The fetch worker died, the main
process waited forever for a batch that would never arrive, and the run deadlocked with the GPU at
0% -- 69 minutes before it was noticed.

THE SUSPECTED MISMATCH, two different definitions of "first":
  * data_processing.add_queries creates a query row for every row with `is_first_review == False`,
    and `is_first_review` IS `elapsed_days == -1`;
  * insert_probes excludes only the card's FIRST OCCURRENCE WITHIN THE CHUNK.
Those agree only if each card has AT MOST ONE row with elapsed_days == -1. A second such row is not
the in-chunk first occurrence, so it is ELIGIBLE as a probe target, yet it got no query row.

WHY IT WOULD BE NEW IN THE -id PIPELINE. `build_parquet_id.py` recomputes elapsed_days from the
CORRECTED show-time, and the frame is sorted by review_time only afterwards, so the correction can
REORDER two adjacent reviews of one card (CLAUDE.md records this as the origin of the NaN landmine).
A reorder can leave the -1 on a row that is no longer first in time order -- or produce two.

This script counts, per card, how many rows carry elapsed_days == -1, on BOTH datasets, and reports
any card with more than one. Published is the control: featA ran 21,870 steps on it without ever
hitting this.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/diag_query_gap.py [n_users]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PUB = Path(r"C:\Users\Andrew\anki-revlogs-10k")
IDD = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 40
stride = max(1, 5000 // n_users)
uids = list(range(1, 5001, stride))[:n_users]


def scan(root, uid, sort_by_time):
    d = root / "revlogs" / ("user_id=%d" % uid)
    if not d.exists():
        return None
    df = pd.read_parquet(d)
    if len(df) < 2:
        return None
    if sort_by_time and "review_time" in df.columns:
        df = df.sort_values("review_time", kind="stable").reset_index(drop=True)
    first = (df["elapsed_days"] == -1)
    per_card = first.groupby(df["card_id"]).sum()
    multi = per_card[per_card > 1]
    # Also: is the -1 row actually the card's FIRST row in the frame's order?
    order_ok = True
    if len(multi) == 0:
        idx_first_row = df.groupby("card_id").head(1).index
        flagged = set(df.index[first])
        # every flagged row should be a card's first row in frame order
        order_ok = flagged.issubset(set(idx_first_row))
    return dict(rows=len(df), cards=int(df["card_id"].nunique()),
                n_first=int(first.sum()), n_multi=int(len(multi)),
                extra=int(multi.sum() - len(multi)) if len(multi) else 0,
                order_ok=order_ok)


print("%-6s | %-34s | %-34s" % ("user", "PUBLISHED", "-id"))
print("%-6s | %-34s | %-34s" % ("", "rows  cards  -1rows multi extra", "rows  cards  -1rows multi extra"))
tot_pub_multi = tot_id_multi = 0
bad_order = []
for uid in uids:
    a = scan(PUB, uid, False)
    b = scan(IDD, uid, True)
    if a is None or b is None:
        continue
    tot_pub_multi += a["n_multi"]
    tot_id_multi += b["n_multi"]
    if not b["order_ok"]:
        bad_order.append(uid)
    flag = "  <== " if (b["n_multi"] > a["n_multi"]) else ""
    print("%-6d | %6d %6d %6d %5d %5d | %6d %6d %6d %5d %5d%s"
          % (uid, a["rows"], a["cards"], a["n_first"], a["n_multi"], a["extra"],
             b["rows"], b["cards"], b["n_first"], b["n_multi"], b["extra"], flag))

print("")
print("cards with MORE THAN ONE elapsed_days == -1 row:")
print("  published : %d" % tot_pub_multi)
print("  -id       : %d" % tot_id_multi)
if bad_order:
    print("  -id users where a -1 row is NOT the card's first row in time order: %s"
          % bad_order[:10])
print("")
if tot_id_multi > tot_pub_multi:
    print("CONFIRMED: the -id set produces cards with a SECOND elapsed_days == -1 row.")
    print("Such a row gets NO query row (add_queries filters on is_first_review) but IS eligible")
    print("as a probe target (insert_probes excludes only the in-chunk first occurrence), so")
    print("insert_probes raises KeyError and takes the fetch worker with it.")
elif tot_id_multi == tot_pub_multi == 0:
    print("NOT CONFIRMED by this mechanism -- neither set has a duplicate first-review row.")
    print("The KeyError must come from somewhere else; look at chunk boundaries next.")
else:
    print("INCONCLUSIVE: both sets show duplicates (%d vs %d). If published also has them, the"
          % (tot_pub_multi, tot_id_multi))
    print("bug is latent in BOTH and featA was merely lucky -- check that before blaming the swap.")
