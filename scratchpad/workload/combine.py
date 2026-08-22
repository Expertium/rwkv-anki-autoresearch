"""Turn per-review intervals into a per-calendar-day workload, and the two arms into a
ratio.

THE WORKLOAD MODEL (Andrew 2026-08-21). After every review an algorithm would assign an
interval; a card on an interval of L days is seen 1/L times per day on average; so the
collection's workload on day D is

    W(D) = sum over cards active on D of 1 / interval_assigned_at_that_card's_last_review

and the efficiency estimate is the ratio W_FSRS(D) / W_RWKV(D), averaged over days. Lower
workload at the same desired retention = more efficient.

HOW IT IS COMPUTED. Review j of card c, on day d_j, with next review of the same card on
day d_next_j, contributes 1/ivl_j to every day in [d_j, d_next_j). That is an interval-add
over a day axis, so one difference array plus a cumsum gives W(D) for EVERY day at once --
no per-day loop, and no reason to subsample to every 50th day to save time.

TWO ACTIVITY DEFINITIONS, because they answer different questions:
  alive   (default) -- a card counts only BETWEEN two observed reviews. A card whose last
                       observed review has passed leaves the collection. This is the
                       closest thing to the cards actually in rotation.
  persist           -- a card counts from its first review to the end of history, i.e.
                       abandoned cards keep contributing forever. Reported as a
                       sensitivity check; it inflates both arms.

THE FLOOR. sum(1/interval) is dominated by whatever produces the SHORTEST intervals, and
at DR=99% FSRS-7 asks for sub-second intervals on a large fraction of rows (a known
pathology -- srs-benchmark's README describes the "sched. penalties" variant as fixing
"extremely short intervals for same-day reviews at high (97-99%) desired retention").
A scheduler inside Anki cannot act on those: a review card's interval is at least a day.
So the headline number floors intervals at 1 day, and the unfloored version is reported
beside it with the floored fraction, rather than one being hidden.

Usage:
  .venv/Scripts/python.exe scratchpad/workload/combine.py <out_dir> [--floor-days 1.0]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DR_LEVELS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
DR_COLS = ["ivl_%d" % round(dr * 100) for dr in DR_LEVELS]


def daily_workload(df, col, n_days, floor_days, mode, only_floored=False, row_mask=None):
    """W(D) for every day 0..n_days-1, from one arm's per-review intervals.

    df must be sorted chronologically and carry card_id / day_offset / <col>.

    only_floored=True restricts the sum to rows the FLOOR actually rewrote. Its share of
    the total is the honest health check on the high-DR rows: when most of the workload
    comes from intervals that were clipped to one day, the ratio at that DR is measuring
    the floor, not the two algorithms, and the numbers converge on 1.0 for that reason
    alone.

    row_mask drops rows from BOTH the workload and the active-card count -- it selects
    which scheduling decisions are counted at all. It must be identical for both arms (it
    is derived from the shared review table, not from either arm's intervals), or the two
    workloads stop being a ratio of anything.
    """
    day = df["day_offset"].to_numpy(dtype=np.int64)
    card = df["card_id"].to_numpy()
    ivl = df[col].to_numpy(dtype=np.float64)
    was_floored = ivl < floor_days if floor_days > 0 else np.zeros(len(ivl), dtype=bool)
    # Hard lower bound at one second even when the caller asks for no floor. Both arms'
    # inversions already bottom out there, but the ACTUAL-schedule reference arm can carry a
    # gap of exactly 0 (a same-day next review), and 1/0 = inf would poison the cumsum for
    # every later day rather than just that row -- even though the row is masked out
    # afterwards, since inf - inf is nan.
    ivl = np.maximum(ivl, max(floor_days, 1.0 / 86400.0))
    rate = 1.0 / ivl
    if only_floored:
        rate = np.where(was_floored, rate, 0.0)

    # next review day of the SAME card; sentinel = n_days (runs to the end of history)
    order = np.lexsort((day, card))
    nxt = np.full(len(df), n_days, dtype=np.int64)
    c_sorted, d_sorted = card[order], day[order]
    same = c_sorted[:-1] == c_sorted[1:]
    tmp = np.full(len(df), n_days, dtype=np.int64)
    tmp[:-1] = np.where(same, d_sorted[1:], n_days)
    nxt[order] = tmp
    if mode == "alive":
        # a card's LAST review ends its life: it contributes nothing afterwards
        is_last = np.zeros(len(df), dtype=bool)
        tmp_last = np.zeros(len(df), dtype=bool)
        tmp_last[:-1] = ~same
        tmp_last[-1] = True
        is_last[order] = tmp_last
        end = np.where(is_last, day, nxt)
    elif mode == "persist":
        end = nxt.copy()
        # the last review of a card runs to the end of history (nxt is already n_days)
    else:
        raise ValueError(mode)

    diff = np.zeros(n_days + 1, dtype=np.float64)
    cnt = np.zeros(n_days + 1, dtype=np.float64)
    lo = np.clip(day, 0, n_days)
    hi = np.clip(end, 0, n_days)
    keep = hi > lo
    if row_mask is not None:
        keep = keep & np.asarray(row_mask, dtype=bool)
    np.add.at(diff, lo[keep], rate[keep])
    np.add.at(diff, hi[keep], -rate[keep])
    np.add.at(cnt, lo[keep], 1.0)
    np.add.at(cnt, hi[keep], -1.0)
    return np.cumsum(diff)[:n_days], np.cumsum(cnt)[:n_days]


def user_ratios(fsrs, rwkv, floor_days, mode, min_cards, day_step):
    n_days = int(max(fsrs["day_offset"].max(), rwkv["day_offset"].max())) + 1
    rows = []
    for dr, col in zip(DR_LEVELS, DR_COLS):
        wf, cf = daily_workload(fsrs, col, n_days, floor_days, mode)
        wr, cr = daily_workload(rwkv, col, n_days, floor_days, mode)
        # both arms replay the same rows, so the active-card count must agree exactly;
        # if it does not, the two workloads are over different card sets and the ratio
        # is not a ratio of anything.
        assert np.allclose(cf, cr), "active-card counts differ between arms"
        sel = (cf >= min_cards) & (wr > 0) & (wf > 0)
        if day_step > 1:
            step_mask = np.zeros(n_days, dtype=bool)
            step_mask[::day_step] = True
            sel &= step_mask
        if not sel.any():
            rows.append({"dr": dr, "n_days": 0})
            continue
        ratio = wf[sel] / wr[sel]
        rows.append({
            "dr": dr,
            "n_days": int(sel.sum()),
            "mean_ratio": float(ratio.mean()),
            "median_ratio": float(np.median(ratio)),
            "pooled_ratio": float(wf[sel].sum() / wr[sel].sum()),
            "mean_w_fsrs": float(wf[sel].mean()),
            "mean_w_rwkv": float(wr[sel].mean()),
            "mean_active_cards": float(cf[sel].mean()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--floor-days", type=float, default=1.0)
    ap.add_argument("--mode", default="alive", choices=["alive", "persist"])
    ap.add_argument("--min-cards", type=int, default=20)
    ap.add_argument("--day-step", type=int, default=1)
    ap.add_argument("--fsrs-prefix", default="fsrs")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    per_user = {}
    for f in sorted(out.glob("%s_u*.parquet" % args.fsrs_prefix)):
        uid = int(f.stem.split("_u")[1])
        r = out / ("rwkv_u%d.parquet" % uid)
        if not r.exists():
            continue
        fsrs, rwkv = pd.read_parquet(f), pd.read_parquet(r)
        assert len(fsrs) == len(rwkv), "arm length mismatch for user %d" % uid
        assert (fsrs["review_th"].to_numpy() == rwkv["review_th"].to_numpy()).all()
        assert (fsrs["card_id"].to_numpy() == rwkv["card_id"].to_numpy()).all()
        per_user[uid] = user_ratios(fsrs, rwkv, args.floor_days, args.mode,
                                    args.min_cards, args.day_step)

    if not per_user:
        print("no matched user pairs in %s" % out)
        return

    print("users: %d   floor=%.3g d   mode=%s   min_cards=%d   day_step=%d"
          % (len(per_user), args.floor_days, args.mode, args.min_cards, args.day_step))
    print("")
    print("%-5s %10s %10s %10s %12s %12s" % (
        "DR", "mean", "median", "geo-mean", "W_FSRS", "W_RWKV"))
    print("-" * 64)
    summary = []
    for k, dr in enumerate(DR_LEVELS):
        vals = [per_user[u][k] for u in per_user if per_user[u][k].get("n_days", 0) > 0]
        if not vals:
            continue
        m = np.array([v["mean_ratio"] for v in vals])
        med = np.array([v["median_ratio"] for v in vals])
        wf = np.array([v["mean_w_fsrs"] for v in vals])
        wr = np.array([v["mean_w_rwkv"] for v in vals])
        rec = {
            "dr": dr, "n_users": len(vals),
            "mean_of_user_mean_ratio": float(m.mean()),
            "median_of_user_mean_ratio": float(np.median(m)),
            "geomean_of_user_mean_ratio": float(np.exp(np.log(m).mean())),
            "mean_of_user_median_ratio": float(np.median(med)),
            "mean_W_fsrs": float(wf.mean()), "mean_W_rwkv": float(wr.mean()),
        }
        summary.append(rec)
        print("%-5s %10.4f %10.4f %10.4f %12.2f %12.2f" % (
            "%d%%" % round(dr * 100), rec["mean_of_user_mean_ratio"],
            rec["median_of_user_mean_ratio"], rec["geomean_of_user_mean_ratio"],
            rec["mean_W_fsrs"], rec["mean_W_rwkv"]))
    print("")
    print("ratio > 1 = FSRS-7 needs MORE reviews/day at the same desired retention,")
    print("i.e. RWKV-Curve is more efficient by that factor.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "summary": summary,
                       "per_user": {str(u): v for u, v in per_user.items()}}, fh, indent=1)
        print("wrote %s" % args.json_out)


if __name__ == "__main__":
    main()
