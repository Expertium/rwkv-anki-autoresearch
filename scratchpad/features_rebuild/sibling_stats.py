"""Normalization constants + COVERAGE for the sibling features, on the TRAIN half only.

Same methodology as the other 18 constants in `id_features.STATISTICS_ID` (optimization/
feature_stats_id.py): a stride sample across users 1-5000 ONLY. Deriving constants from
5001-10000 would leak eval-set statistics into every candidate's inputs.

Coverage is the point of this script as much as the constants. `preset_age` was DROPPED from the
rebuild for being defined on 1 row in 14, so a new column that is mostly its own sentinel has to
clear that same bar before it earns a slot.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/sibling_stats.py [n_users]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RWKV_ID_FEATURES", "1")
from rwkv.id_features import _log3, _log_t, sibling_gap_seconds  # noqa: E402

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
TIMESTAMP_MIN_MS = 1e11
SHIFT = 1 << 41
LO, HI = 1, 5000
n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 300
stride = max(1, (HI - LO + 1) // n_users)
uids = list(range(LO, HI + 1, stride))[:n_users]

tot = 0
gap_def = 0
g1 = g2 = 0.0
c1 = c2 = 0.0
multi_rows = 0
cnt_hist = np.zeros(12, dtype=np.int64)
cov = []
for i, uid in enumerate(uids):
    d = DATA / "revlogs" / f"user_id={uid}"
    if not d.exists():
        continue
    r = pd.read_parquet(d)
    if len(r) < 2:
        continue
    c = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", uid)])
    if "user_id" in c.columns:
        c = c.drop(columns=["user_id"])
    df = r.merge(c, on="card_id", how="left", validate="many_to_one")
    df = df.sort_values("review_time", kind="stable").reset_index(drop=True)

    g = sibling_gap_seconds(df)
    ok = g >= 0.0
    tot += g.size
    gap_def += int(ok.sum())
    cov.append(float(ok.mean()))
    if ok.any():
        v = _log_t(g[ok])
        g1 += float(v.sum())
        g2 += float((v * v).sum())

    # --- sibling COUNT as of review time, VECTORIZED. card_id IS a creation timestamp here,
    # so "cards of this note created before this review" is computable without leaking cards the
    # user goes on to add later. Composite int64 key (dense note index << 41 | timestamp) turns
    # the per-row sibling search into two searchsorted calls; the row-loop version was ~40x slower
    # and was what made the first 300-user attempt unusable.
    nid = pd.to_numeric(df["note_id"], errors="coerce").to_numpy(dtype=np.float64)
    cid = pd.to_numeric(df["card_id"], errors="coerce").to_numpy(dtype=np.float64)
    rt = df["review_time"].to_numpy(dtype=np.int64)
    cn = pd.to_numeric(c["note_id"], errors="coerce").to_numpy(dtype=np.float64)
    cc = pd.to_numeric(c["card_id"], errors="coerce").to_numpy(dtype=np.float64)
    okc = np.isfinite(cn) & np.isfinite(cc) & (cc >= TIMESTAMP_MIN_MS)
    cn, cc = cn[okc], cc[okc]
    if cn.size:
        uniq, ci = np.unique(cn, return_inverse=True)
        key_card = np.sort(ci.astype(np.int64) * SHIFT + cc.astype(np.int64))
        ri = np.searchsorted(uniq, nid)
        good = np.isfinite(nid) & (ri < uniq.size)
        ri = np.clip(ri, 0, max(uniq.size - 1, 0))
        good &= uniq[ri] == np.nan_to_num(nid, nan=-1.0)
        base = ri.astype(np.int64) * SHIFT
        cnt = (np.searchsorted(key_card, base + rt, side="right")
               - np.searchsorted(key_card, base, side="left")).astype(np.float64)
        cnt = np.where(good, cnt, 0.0)
        cnt -= np.where(good & (cid < rt.astype(np.float64)), 1.0, 0.0)  # exclude self
        cnt = np.maximum(cnt, 0.0)
        multi_rows += int((cnt > 0).sum())
        np.add.at(cnt_hist, np.minimum(cnt.astype(np.int64), 11), 1)
        v = _log3(cnt)
        c1 += float(v.sum())
        c2 += float((v * v).sum())

    if (i + 1) % 50 == 0:
        print("  %d users, %d rows, gap cov %.4f, sib>0 %.4f"
              % (i + 1, tot, gap_def / max(tot, 1), multi_rows / max(tot, 1)), flush=True)

gm = g1 / gap_def
gs = float(np.sqrt(max(g2 / gap_def - gm * gm, 0.0)))
cm = c1 / tot
cs = float(np.sqrt(max(c2 / tot - cm * cm, 0.0)))
print("")
print("--- %d users / %d rows" % (len(cov), tot))
print("GAP   defined rows : %d (%.4f)   per-user median/p10/p90 %.4f / %.4f / %.4f"
      % (gap_def, gap_def / tot, np.median(cov), np.percentile(cov, 10),
         np.percentile(cov, 90)))
print("COUNT rows with >=1 prior sibling card : %d (%.4f)" % (multi_rows, multi_rows / tot))
print("count histogram 0..10,11+ : %s" % (cnt_hist.tolist(),))
print("")
print('    "sibling_gap_mean": %.4f,' % gm)
print('    "sibling_gap_std": %.4f,' % gs)
print('    "sibling_count_mean": %.4f,' % cm)
print('    "sibling_count_std": %.4f,' % cs)
