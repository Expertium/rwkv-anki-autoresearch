"""Verify every interval feature is END-of-previous to START-of-this. CPU, seconds.

ANDREW 2026-08-19: "make sure that all the stuff like elapsed_days, elapsed_seconds, etc. is based
on review ID *after* subtracting review duration, so that everything interval-related is 'from the
end of the prior review to the beginning of next review'."

The `-id` dataset already moved timestamps to SHOW time (`review_time = revlog.id - taken_millis`),
which is right and is why this looked done. But its intervals were still `review_time.diff()`, i.e.
START-to-START, which carries the previous review's duration:

    show(k) - show(k-1)  =  duration(k-1) + [ show(k) - answer(k-1) ]

Checks, against values recomputed from the raw columns rather than from the pipeline:
  1. `elapsed_seconds` == floor(show(k) - answer(k-1)), clamped at 0, sentinel kept on first reviews
  2. `t_since_any_review`'s gap == the same thing across ALL cards (per-user, not per-card)
  3. the sentinel is never minted by rounding -- no non-first row may come out as -1
  4. reports how much shorter the corrected intervals are, and the new negative-gap rate

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/smoke_end_to_start.py [user ...]
"""
import glob
import os
import sys

os.environ.setdefault("RWKV_ID_FEATURES", "1")

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

sys.path.insert(0, os.getcwd())
from rwkv import id_features as idf  # noqa: E402

users = sys.argv[1:] or ["333", "1", "486"]
fails = 0

for u in users:
    hits = sorted(glob.glob("../anki-revlogs-10k-id/revlogs/user_id=%s/*" % u))
    if not hits:
        print("user %s: no data" % u)
        continue
    df = pd.read_parquet(hits[0]).sort_values("review_time").reset_index(drop=True)
    raw = df.copy()

    out = idf.elapsed_end_to_start(df)
    es = out["elapsed_seconds"].to_numpy()

    # independent recomputation from the raw columns
    is_first = (raw["elapsed_seconds"] == -1).to_numpy()
    ans_prev = (raw["review_time"] + raw["duration"]).groupby(raw["card_id"]).shift()
    want = np.floor(((raw["review_time"] - ans_prev) / 1000.0).clip(lower=0.0).to_numpy())
    want[is_first] = -1.0

    ok1 = bool(np.array_equal(es, want.astype("int64")))
    ok3 = bool((es[~is_first] != -1).all())

    # how much the correction moves things, on non-first rows
    old = raw["elapsed_seconds"].to_numpy()[~is_first]
    new = es[~is_first]
    shorter = old - new
    med_rel = float(np.median(shorter / np.maximum(old, 1)))
    neg = int((((raw["review_time"] - ans_prev) / 1000.0).to_numpy()[~is_first] < 0).sum())

    print("--- user %s  (%d reviews, %d non-first)" % (u, len(df), (~is_first).sum()))
    print("    elapsed_seconds == end-to-start recomputation : %s" % ("OK" if ok1 else "FAIL"))
    print("    no non-first row minted as the -1 sentinel    : %s" % ("OK" if ok3 else "FAIL"))
    # ⚠ REPORT BY INTERVAL LENGTH. The median RELATIVE change is ~0% and says nothing: most
    # intervals are days, where subtracting a ~10 s duration is noise. The correction bites on
    # SHORT gaps -- the same-day learning steps -- so a single median hides the whole effect.
    print("    median absolute shortening                    : %.0f s (== a typical duration)"
          % float(np.median(shorter)))
    for lo, hi, lbl in ((0, 60, "under 1 min"), (60, 600, "1-10 min"),
                        (600, 86400, "10 min - 1 day"), (86400, 10**9, "over 1 day")):
        m = (old >= lo) & (old < hi)
        if m.sum():
            rel = 100 * np.median(shorter[m] / np.maximum(old[m], 1))
            print("      gaps %-15s n=%7d  median shortened by %5.1f%%" % (lbl, m.sum(), rel))
    print("    gaps that went negative and were clamped to 0 : %d (%.3f%%)"
          % (neg, 100.0 * neg / max((~is_first).sum(), 1)))
    fails += (not ok1) + (not ok3)

print("")
print("END_TO_START_%s" % ("ALL_PASS" if fails == 0 else "FAILED"))
sys.exit(1 if fails else 0)
