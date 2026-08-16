#!/usr/bin/env python
"""ON vs OFF on the same `-id` data: is the feature change PURELY ADDITIVE? CPU, ~1 min.

The de-risk build's whole point is a control the existing DB cannot provide: the published-vs-`-id`
dataset swap alone moves `size` for ~30% of users, so any difference measured against `train_db_5k_h1`
would be unattributable. Both arms here read the same dataset, users, batch size and label filter --
only the column set differs.

THE THREE CLAIMS, in increasing order of what they would catch:

 1. **Identical row structure.** Same users, same `{uid}_batches` chunk lists, same per-chunk row
    counts. A change here means the new derivations perturbed segmentation or chunking, which would
    invalidate the `size` gate for reasons that have nothing to do with the features.
 2. **★ The 23 shared columns are BIT-IDENTICAL.** This is the real prize: it proves the change is
    additive rather than a rewrite of the input the model has been trained on for months. It is also
    the check that would catch the clamp doing more than intended -- the clamp touches
    `elapsed_seconds`, which feeds four of those 23 columns, so ANY row it fires on shows up here.
    Expect a small number of legitimately differing rows for exactly that reason; they are listed
    rather than tolerated silently.
 3. **The 21 new columns are alive.** Finite, and non-degenerate (a column that is one constant
    everywhere is a derivation bug, not a feature).

ASCII output only.
"""
import io
import json
import sys

import lmdb
import numpy as np
import torch

OFF = "F:/rwkv_lmdb/derisk_id_off"
ON = "F:/rwkv_lmdb/derisk_id_on"
# The one column the rebuild removes; every other OFF column must survive unchanged.
DROPPED = "scaled_state"
OFF_COLS = [
    "scaled_elapsed_days", "scaled_elapsed_days_cumulative", "scaled_elapsed_seconds",
    "elapsed_seconds_sin", "elapsed_seconds_cos", "scaled_elapsed_seconds_cumulative",
    "elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos", "scaled_duration",
    "rating_1", "rating_2", "rating_3", "rating_4", "note_id_is_nan", "deck_id_is_nan",
    "preset_id_is_nan", "day_offset_diff", "day_of_week", "diff_new_cards", "diff_reviews",
    "cum_new_cards_today", "cum_reviews_today", DROPPED, "is_query",
]
NEW_COLS_START = len(OFF_COLS) - 1  # ON = OFF minus scaled_state, then the new columns appended

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def load(txn, key):
    raw = txn.get(key.encode())
    return None if raw is None else torch.load(io.BytesIO(raw), weights_only=True)


def main():
    eoff = lmdb.open(OFF, readonly=True, lock=False, subdir=True,
                     map_size=30_000_000_000, max_readers=2048)
    eon = lmdb.open(ON, readonly=True, lock=False, subdir=True,
                    map_size=30_000_000_000, max_readers=2048)
    with eoff.begin() as toff, eon.begin() as ton:
        users = []
        for uid in range(1, 101):
            a = toff.get(f"{uid}_batches".encode())
            b = ton.get(f"{uid}_batches".encode())
            if a is None and b is None:
                continue
            if (a is None) != (b is None):
                fails.append(f"user {uid} present in only one arm")
                continue
            if json.loads(a) != json.loads(b):
                fails.append(f"user {uid} chunk list differs")
                continue
            users.append((uid, json.loads(a)))
        check("both arms hold the same users with identical chunk lists", not fails,
              f"{len(users)} users")

        n_rows = 0
        worst = 0.0
        worst_where = ""
        diff_rows = 0
        newstats = None
        for uid, keys in users:
            for s_, e_, L in keys:
                pre = f"{uid}_{s_}-{e_}_{L}_card_features"
                fa = load(toff, pre)
                fb = load(ton, pre)
                if fa is None or fb is None:
                    fails.append(f"missing card_features for {pre}")
                    continue
                if fa.shape[0] != fb.shape[0]:
                    fails.append(f"row count differs at {pre}: {fa.shape} vs {fb.shape}")
                    continue
                n_rows += fa.shape[0]
                a = fa.float().numpy()
                b = fb.float().numpy()
                # the 23 shared columns: OFF's list minus scaled_state, in order, is exactly ON's
                # first 23 columns
                keep_idx = [i for i, c in enumerate(OFF_COLS) if c != DROPPED]
                d = np.abs(a[:, keep_idx] - b[:, :len(keep_idx)])
                m = float(d.max()) if d.size else 0.0
                diff_rows += int((d.max(axis=1) > 0).sum()) if d.size else 0
                if m > worst:
                    worst, worst_where = m, pre
                nb = b[:, NEW_COLS_START:]
                st = np.vstack([nb.min(axis=0), nb.max(axis=0), nb.std(axis=0)])
                newstats = st if newstats is None else np.vstack(
                    [np.minimum(newstats[0], st[0]), np.maximum(newstats[1], st[1]),
                     np.maximum(newstats[2], st[2])]
                )

        check("widths are 24 and 44", fa.shape[1] == 24 and fb.shape[1] == 44,
              f"{fa.shape[1]} / {fb.shape[1]}")
        check("the 23 shared columns are BIT-IDENTICAL", worst == 0.0,
              f"max |delta| {worst:.3e} at {worst_where or 'none'}; "
              f"{diff_rows} of {n_rows:,} rows differ")
        if newstats is not None:
            check("new columns finite", bool(np.isfinite(newstats).all()))
            check("no new column is constant everywhere", bool((newstats[2] > 1e-9).all()),
                  f"min std across new columns {float(newstats[2].min()):.4g}")
        print(f"\n  compared {n_rows:,} rows over {len(users)} users")

    print("\n" + ("DERISK_ALL_PASS" if not fails else "DERISK_FAILURES: " + "; ".join(fails[:8])))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
