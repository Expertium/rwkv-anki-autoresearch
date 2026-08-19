"""Verify the cumulative-elapsed sentinel fix on real data (CPU, seconds).

THE BUG (reported by obezag on Discord 2026-08-19, confirmed in code). A card's first review carries
`elapsed_days`/`elapsed_seconds` = -1, a SENTINEL meaning "no previous review". The old derivation was
a plain `groupby("card_id").cumsum()`, so it summed the sentinel as a magnitude and every cumulative
value was low by exactly 1.

WHY IT IS WORSE THAN AN OFF-BY-ONE. The feature is `where(x == -1, 0, log(1 + 1e-5 + x))`, so storing
`C - 1` gives `log(C)` where `log(1 + C)` is meant. At C = 1 that is `log(1) = 0` -- EXACTLY the value
the sentinel encodes. A card's second review one day after its first was indistinguishable, on this
feature, from a review with no history at all.

Three checks, each of which the old code fails:
  1. the sentinel SURVIVES on first reviews (so they still take the `x == -1` branch);
  2. non-first cumulative equals the TRUE sum of real gaps;
  3. the sentinel COLLISION is gone -- no non-first review encodes to the sentinel's feature value.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/smoke_cumsum_sentinel.py [user]
"""
import glob
import sys

import numpy as np
import pandas as pd

user = sys.argv[1] if len(sys.argv) > 1 else "333"
path = sorted(glob.glob("../anki-revlogs-10k-id/revlogs/user_id=%s/*" % user))[0]
df = pd.read_parquet(path)
if "review_time" in df:
    df = df.sort_values("review_time").reset_index(drop=True)

df["is_first_review"] = (df["elapsed_days"] == -1).astype(int)
is_first = df["is_first_review"].astype(bool)

def old(col):
    return df.groupby("card_id")[col].cumsum().to_numpy()

def new(col):
    gap = df[col].where(~is_first, 0)
    return gap.groupby(df["card_id"]).cumsum().where(~is_first, -1).to_numpy()

def feat(x):
    return np.where(x == -1, 0.0, np.log(1 + 1e-5 + np.maximum(x, -0.99)))

fails = 0
for col in ("elapsed_days", "elapsed_seconds"):
    o, n = old(col), new(col)
    true_cum = df[col].where(~is_first, 0).groupby(df["card_id"]).cumsum().to_numpy()

    ok1 = bool((n[is_first.to_numpy()] == -1).all())
    ok2 = bool(np.array_equal(n[~is_first.to_numpy()], true_cum[~is_first.to_numpy()]))
    # ⚠ COUNT THE MEANINGFUL COLLISION ONLY. An exact-zero test lumps two classes together and
    # FLATTERS the fix: rows whose old cumsum was -1 (every gap so far was 0, e.g. same-day
    # learning steps) encode to 0 under the old code and to log(1+1e-5) ~ 1e-5 under the new one --
    # numerically the same, so nothing is actually repaired there. The class that IS repaired is
    # old cumsum == 0, i.e. a TRUE cumulative of exactly 1: old gives log(1) = 0 (the sentinel's
    # own value) and new gives log(2) = 0.693, a 0.31 sigma move.
    nf = ~is_first.to_numpy()
    coll_old = int((o[nf] == 0).sum())          # true cumulative 1 day, encoded as "no history"
    coll_new = int((n[nf] == -1).sum())         # must be zero: -1 is reserved for first reviews
    ok3 = coll_new == 0
    d = np.abs(feat(n) - feat(o))[nf] / 2.25    # normalized by the column's std

    print("--- %s" % col)
    print("    sentinel preserved on first reviews : %s" % ("OK" if ok1 else "FAIL"))
    print("    non-first == true sum of real gaps  : %s" % ("OK" if ok2 else "FAIL"))
    print("    rows whose TRUE cum of 1 read as 'no history': %d (%.1f%% of non-first)"
          % (coll_old, 100.0 * coll_old / max(nf.sum(), 1)))
    print("    -1 reserved for first reviews only  : %s" % ("OK" if ok3 else "FAIL"))
    print("    normalized feature shift: mean %.4f  p90 %.4f  max %.4f sigma"
          % (d.mean(), np.percentile(d, 90), d.max()))
    fails += (not ok1) + (not ok2) + (not ok3)

print("")
print("CUMSUM_SENTINEL_%s" % ("ALL_PASS" if fails == 0 else "FAILED"))
sys.exit(1 if fails else 0)
