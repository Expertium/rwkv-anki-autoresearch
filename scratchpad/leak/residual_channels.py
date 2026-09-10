"""Which leak channels does RWKV_CLOCK_AT_PREV_ANSWER leave open? (2026-09-10, before the leak-free WS)

The fix moves an in-session query/probe row back by delta = its gap since the previous answer on any
card, and shifts every CONTINUOUS clock column. Three kinds of column move only when the shift crosses
something, and the fix does not touch them. This counts how often each one would change, on the raw
-id parquet (read only), for the rows the fix applies to: non-first reviews with 0 < gap < T.
  1. UTC-day columns   -- dow/doy/is_weekend and the review-time real cycles (UTC day index)
  2. Anki-day columns  -- elapsed_days (+cumulative), day_offset_diff, cum_*_today (the rollover day)
  3. creation-batch counts, clipped at the show time: a card created inside the shift window is
     counted at the reconstructed show time and would not be at the previous answer.
Usage: python scratchpad/leak/residual_channels.py [users...]   (default 5001-5040)
"""
import sys

import numpy as np
import pandas as pd

ROOT = "C:/Users/Andrew/anki-revlogs-10k-id"
DAY_MS = 86_400_000
TS_MIN = 1e11
users = [int(u) for u in sys.argv[1:]] or list(range(5001, 5041))
WIN = {"1min": 60_000, "1h": 3_600_000, "1d": 86_400_000}
tot = {}


def add(k, v):
    tot[k] = tot.get(k, 0) + int(v)


for u in users:
    try:
        r = pd.read_parquet(f"{ROOT}/revlogs/user_id={u}/data.parquet",
                            columns=["review_time", "card_id", "day_offset", "duration", "state"])
        c = pd.read_parquet(f"{ROOT}/cards/user_id={u}/data.parquet", columns=["card_id"])
    except Exception:
        continue
    r = r.sort_values("review_time", kind="stable").reset_index(drop=True)
    if len(r) < 100:
        continue
    rt = r["review_time"].to_numpy(np.int64)
    du = r["duration"].to_numpy(np.int64)
    do = r["day_offset"].to_numpy(np.int64)
    cid = r["card_id"].to_numpy(np.float64)
    first = r["state"].to_numpy() == 0
    gap = np.full(len(rt), -1.0)
    gap[1:] = np.maximum((rt[1:] - (rt[:-1] + du[:-1])) / 1000.0, 0.0)
    q = ~first
    q[0] = False
    add("query_rows", q.sum())
    cards = np.sort(c["card_id"].to_numpy(np.float64))
    cards = cards[np.isfinite(cards) & (cards >= TS_MIN)]
    for T in (1800, 10800):
        m = q & (gap > 0) & (gap < T)
        d_ms = np.where(m, gap * 1000.0, 0.0)
        add(f"T{T}_shifted", m.sum())
        utc = m & (np.floor(rt / DAY_MS) != np.floor((rt - d_ms) / DAY_MS))
        add(f"T{T}_utc_cross", utc.sum())
        dprev = np.r_[do[0], do[:-1]]
        anki = m & (do != dprev)
        add(f"T{T}_anki_cross", anki.sum())
        ts = cid >= TS_MIN
        anyc = np.zeros(len(rt), bool)
        for lab, w in WIN.items():
            hi_s = np.minimum(cid + w, rt.astype(np.float64))
            hi_f = np.minimum(cid + w, rt - d_ms)
            n_s = np.searchsorted(cards, hi_s, side="right")
            n_f = np.searchsorted(cards, hi_f, side="right")
            ch = m & ts & (n_s != n_f)
            anyc |= ch
            add(f"T{T}_batch_{lab}", ch.sum())
        add(f"T{T}_batch_any", anyc.sum())
        add(f"T{T}_any_channel", (utc | anki | anyc).sum())
    add("users", 1)

n = tot["query_rows"]
print(f"users {tot['users']}  query rows (non-first reviews) {n:,}")
for T in (1800, 10800):
    s = tot[f"T{T}_shifted"]
    print(f"\nT={T} s: rows the fix shifts {s:,} ({s / n:.4%} of query rows)")
    for k in ("utc_cross", "anki_cross", "batch_1min", "batch_1h", "batch_1d", "batch_any", "any_channel"):
        v = tot[f"T{T}_{k}"]
        print(f"  {k:12s} {v:8,d}   {v / n:.5%} of query rows   {v / max(s, 1):.4%} of shifted rows")
