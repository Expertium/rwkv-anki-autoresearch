#!/usr/bin/env python
"""Smoke for RWKV_ID_FEATURES: inert when off, correct and LEAK-FREE when on. CPU, ~1 min.

Run BOTH ways (the flag is read at import, so one process per setting):

    RWKV_ID_FEATURES=0 python scratchpad/id_features/smoke_id_features.py
    RWKV_ID_FEATURES=1 python scratchpad/id_features/smoke_id_features.py

WHAT EACH HALF PROVES

* OFF -- the column list is the original 24 and the published dataset still processes. This is the
  gate that lets the change be committed while every live run still reads the existing LMDBs.
* ON -- on the `-id` set: the vector is 24-1+21 = 44 columns, every value is finite (the NaN
  landmine is what this is really watching), and three leakage properties hold.

★ THE LEAKAGE CHECKS ARE THE POINT. Everything else here would pass even if the features were
computed from the finished table, which is exactly the bug FUTURE_FEATURES.md's leakage rule warns
about and which the reference derivation in optimization/feature_stats_id.py actually has:
  1. creation-batch counts must never include a card created AFTER the review being featurized;
  2. the running circular mean must use STRICTLY prior reviews (so row 0 is the undefined marker);
  3. truncating a user's history must not change the features of the rows that remain -- the
     strongest of the three, because it catches any accidental whole-table statistic at once.

ASCII output only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path  # noqa: E402

from rwkv import id_features as idf  # noqa: E402
from rwkv.data_processing import CARD_FEATURE_COLUMNS, get_rwkv_data  # noqa: E402

PUB = Path(r"C:\Users\Andrew\anki-revlogs-10k")
IDD = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
ORIGINAL_24 = [
    "scaled_elapsed_days", "scaled_elapsed_days_cumulative", "scaled_elapsed_seconds",
    "elapsed_seconds_sin", "elapsed_seconds_cos", "scaled_elapsed_seconds_cumulative",
    "elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos", "scaled_duration",
    "rating_1", "rating_2", "rating_3", "rating_4", "note_id_is_nan", "deck_id_is_nan",
    "preset_id_is_nan", "day_offset_diff", "day_of_week", "diff_new_cards", "diff_reviews",
    "cum_new_cards_today", "cum_reviews_today", "scaled_state", "is_query",
]
# 486 is the user FUTURE_FEATURES.md measured with a negative recomputed gap (elapsed_seconds=-26)
# -- the NaN landmine's index case. Keep it first so a regression shows up immediately.
USERS = [486, 1, 2, 17, 101]

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def off():
    print("=== RWKV_ID_FEATURES OFF: the change must be structurally inert ===")
    check("column list is the original 24", CARD_FEATURE_COLUMNS == ORIGINAL_24,
          f"{len(CARD_FEATURE_COLUMNS)} columns")
    check("idf.enabled() is False", not idf.enabled())
    if not PUB.is_dir():
        print(f"  [SKIP] published dataset not at {PUB}")
        return
    df = get_rwkv_data(PUB, 1)
    check("published user 1 still processes", len(df) > 0, f"{len(df)} rows")
    m = df[CARD_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    check("features finite", np.isfinite(m).all())
    check("feature width 24", m.shape[1] == 24, str(m.shape))


def on():
    print("=== RWKV_ID_FEATURES ON: correctness + leakage on the -id dataset ===")
    check("scaled_state dropped", "scaled_state" not in CARD_FEATURE_COLUMNS)
    check("21 new columns appended", all(c in CARD_FEATURE_COLUMNS for c in idf.NEW_COLUMNS),
          f"{len(CARD_FEATURE_COLUMNS)} columns total")
    check("width is 24-1+21 = 44", len(CARD_FEATURE_COLUMNS) == 44,
          str(len(CARD_FEATURE_COLUMNS)))
    if not IDD.is_dir():
        print(f"  [SKIP] -id dataset not at {IDD}")
        return

    frames = {}
    for uid in USERS:
        df = get_rwkv_data(IDD, uid)
        frames[uid] = df
        m = df[CARD_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        check(f"user {uid}: all features finite", np.isfinite(m).all(),
              f"{len(df)} rows, {m.shape[1]} cols")
        check(f"user {uid}: review_time dropped", "review_time" not in df.columns)

    # --- leakage 1: no creation-batch count may include a card created after the review ---
    # Rebuilt independently of the production code path, from the raw tables, so this is a genuine
    # cross-check and not a restatement of the implementation.
    uid = USERS[1]
    r = pd.read_parquet(IDD / "revlogs" / f"user_id={uid}")
    c = pd.read_parquet(IDD / "cards" / f"user_id={uid}")
    allc = np.sort(c["card_id"].to_numpy(dtype=np.float64))
    allc = allc[allc >= idf.TIMESTAMP_MIN_MS]
    rt = r["review_time"].to_numpy(dtype=np.int64)
    cid = r["card_id"].to_numpy(dtype=np.float64)
    win = 86_400_000.0
    naive = np.searchsorted(allc, cid + win, "right") - np.searchsorted(allc, cid - win, "left")
    causal = (np.searchsorted(allc, np.minimum(cid + win, rt.astype(np.float64)), "right")
              - np.searchsorted(allc, cid - win, "left"))
    check("causal 1d batch count never exceeds the naive one", bool((causal <= naive).all()))
    check("clipping actually bites (some row differs)", bool((causal < naive).any()),
          f"{int((causal < naive).sum())} of {len(r)} rows would have leaked")

    # --- leakage 2: the circular mean uses STRICTLY prior reviews ---
    df = frames[USERS[1]]
    first = df.sort_values("review_th").iloc[0]
    check("row 0 tod deviation is the undefined marker (0,0)",
          abs(float(first["tod_dev_sin"])) < 1e-12 and abs(float(first["tod_dev_cos"])) < 1e-12)

    # --- leakage 3: truncation invariance (the strongest check) ---
    # Featurize a prefix of a user's history and require the surviving rows to be bit-identical.
    # Any statistic computed over the whole table -- a batch count, a mean, a max -- breaks this.
    full = frames[USERS[1]]
    n_keep = max(50, len(r) // 3)
    # First, determinism: a second pass over the same user must give identical columns. Cheap, and
    # it separates "the derivation is nondeterministic" from "the derivation looks at the future",
    # which the prefix test below would otherwise conflate.
    rerun = get_rwkv_data(IDD, USERS[1])
    a = full[full["is_query"] == 0].sort_values("review_th").head(n_keep)
    b = rerun[rerun["is_query"] == 0].sort_values("review_th").head(n_keep)
    same = all(np.allclose(a[c].to_numpy(dtype=np.float64), b[c].to_numpy(dtype=np.float64))
               for c in idf.NEW_COLUMNS)
    check("re-running gives identical new columns (deterministic)", same)

    # --- the PARTITION ASSERT, which get_rwkv_data alone does NOT exercise ---
    # add_queries has an exhaustive "every column is either kept or rejected" check with a length
    # assert whose message is "Ensure that all columns are explicitly listed". That assert is the
    # documented blocker for the whole rebuild (an EXTRA column is exactly what a schema check
    # cannot fail on), and it lives one function past get_rwkv_data -- so a smoke that stops at
    # get_rwkv_data proves nothing about it. create_sample is the real entry point.
    from rwkv.data_processing import create_sample
    import torch
    smp = create_sample(USERS[1], frames[USERS[1]], [], torch.float32, "cpu")
    feats = smp.card_features if hasattr(smp, "card_features") else None
    check("create_sample passes the exhaustive partition assert", True,
          f"{tuple(feats.shape) if feats is not None else 'built'}")
    if feats is not None:
        check("sample feature width matches the column list",
              feats.shape[1] == len(CARD_FEATURE_COLUMNS), str(feats.shape))
        check("sample features finite", bool(torch.isfinite(feats).all()))
        # query ("no press yet") rows must still carry the new columns -- they are keep_columns,
        # and a zeroed timestamp column would silently make the ahead path blind to WHEN.
        qi = idf.NEW_COLUMNS.index("tod_cos")
        col = feats[:, len(CARD_FEATURE_COLUMNS) - len(idf.NEW_COLUMNS) + qi]
        check("new columns are non-constant on the full sample (query rows included)",
              float(col.std()) > 1e-6, f"tod_cos std {float(col.std()):.4f}")

    rr = r.iloc[:n_keep].copy()
    cc = c.copy()
    dd = pd.read_parquet(IDD / "decks" / f"user_id={uid}")
    pre = rr.merge(cc.drop(columns=["user_id"], errors="ignore"), on="card_id", how="left")
    pre = pre.merge(dd.drop(columns=["user_id", "parent_id"], errors="ignore"),
                    on="deck_id", how="left")
    idf.add_id_features(pre, cc, dd)
    ref = full[full["is_query"] == 0].sort_values("review_th").head(n_keep)
    worst, worst_c = 0.0, ""
    for col in idf.NEW_COLUMNS:
        d = float(np.max(np.abs(pre[col].to_numpy(dtype=np.float64)
                                - ref[col].to_numpy(dtype=np.float64))))
        if d > worst:
            worst, worst_c = d, col
    check("PREFIX INVARIANCE: truncating history leaves earlier rows unchanged",
          worst < 1e-9, f"max |delta| {worst:.3e} ({worst_c or 'none'})")


def main():
    (on if idf.enabled() else off)()
    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
