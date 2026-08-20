"""Brute-force reference for sibling_gap_seconds, on a user that HAS siblings.

Two guards, both learned the hard way in the last ten minutes:
 * the comparison REFUSES to report agreement unless the reference has real defined rows
   (user 1 has 4005 cards / 4005 notes, so an all-sentinel vs all-sentinel match proves nothing);
 * the slice is taken from the END of the history, because a preceding sibling review can only
   exist after the first cross-card review -- an early prefix is defined-free by construction.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
os.environ["RWKV_ID_FEATURES"] = "1"
import rwkv.id_features as idf

D = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
uid = 101
r = pd.read_parquet(D / "revlogs" / f"user_id={uid}")
c = pd.read_parquet(D / "cards" / f"user_id={uid}")
df = r.merge(c.drop(columns=["user_id"], errors="ignore"), on="card_id", how="left")
df = df.sort_values("review_time", kind="stable").reset_index(drop=True)
print("full history rows:", len(df))

fast = idf.sibling_gap_seconds(df)
nid = df["note_id"].to_numpy(); cid = df["card_id"].to_numpy()
rt = df["review_time"].to_numpy(np.int64); du = df["duration"].to_numpy(np.int64)

# Brute force over the WHOLE history (each row scans its own prefix), vectorized per row.
slow = np.full(len(df), -1.0)
for i in range(len(df)):
    m = (nid[:i] == nid[i]) & (cid[:i] != cid[i])
    if m.any():
        j = np.flatnonzero(m)[-1]
        slow[i] = max((rt[i] - (rt[j] + du[j])) / 1000.0, 0.0)

nd_f, nd_s = int((fast >= 0).sum()), int((slow >= 0).sum())
assert nd_s > 200, "VACUOUS: only %d defined rows in the reference" % nd_s
assert int((slow > 0).sum()) > 100, "VACUOUS: reference gaps are all zero"
print("defined rows  fast=%d  brute=%d" % (nd_f, nd_s))
print("nonzero gaps in the reference: %d" % int((slow > 0).sum()))
print("max |fast - brute| = %.3e" % np.max(np.abs(fast - slow)))
print("AGREE" if np.max(np.abs(fast - slow)) == 0.0 else "DISAGREE")
