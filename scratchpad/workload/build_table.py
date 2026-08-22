"""Dump the CANONICAL replay table for one user -- the single row set both arms replay.

WHY THIS EXISTS. FSRS and RWKV have different preprocessing pipelines that keep different
rows (FSRS drops delta_t==0 rows and runs outlier removal; RWKV keeps the raw stream). If
each arm replayed its own row set, the workload ratio would compare two different card
populations and the number would mean nothing. So ONE table is built here, from RWKV's
get_rwkv_data (which is the un-filtered raw stream plus derived columns), and both arms
replay exactly it. The RWKV arm re-derives the same frame and ASSERTS it matches.

Columns are the minimum both arms need. RWKV additionally needs the 92-dim feature row,
which it rebuilds itself from the same source frame.

Usage: .venv/Scripts/python.exe scratchpad/workload/build_table.py <user_id> <out.parquet>
"""
import sys, os
sys.path.insert(0, os.getcwd())
from scratchpad.workload.env_champ import apply
apply()

from pathlib import Path
import pandas as pd

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")
COLS = ["card_id", "review_th", "day_offset", "rating", "elapsed_seconds", "elapsed_days", "i"]


def build(user_id):
    from rwkv.data_processing import get_rwkv_data
    df = get_rwkv_data(DATA, user_id)
    out = df[COLS].copy()
    # get_rwkv_data sorts by (card_id, review_th) internally at some points; the replay
    # needs CHRONOLOGICAL order, and review_th is assigned in file order before any sort.
    out = out.sort_values("review_th", kind="stable").reset_index(drop=True)
    assert out["review_th"].is_monotonic_increasing
    assert out["day_offset"].is_monotonic_increasing
    assert out["rating"].between(1, 4).all()
    # i is the 1-based review index within the card; both arms rely on i==1 meaning
    # "first review of this card" (the FSRS init path, and RWKV's fresh state).
    assert (out.groupby("card_id")["i"].first() == 1).all()
    assert (out.groupby("card_id")["i"].diff().fillna(1) == 1).all()
    return out


if __name__ == "__main__":
    uid = int(sys.argv[1])
    dest = Path(sys.argv[2])
    dest.parent.mkdir(parents=True, exist_ok=True)
    t = build(uid)
    t.to_parquet(dest, index=False)
    print("user %d -> %s  rows=%d cards=%d days=%d..%d"
          % (uid, dest, len(t), t["card_id"].nunique(),
             t["day_offset"].min(), t["day_offset"].max()))
