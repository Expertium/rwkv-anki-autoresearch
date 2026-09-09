"""Read the gen-5 LMDB back and report, per INPUT COLUMN, what the model actually sees.

Reads the stored card_features (what training consumes) rather than re-deriving them, so a
column that is emitted correctly and then dropped, mis-ordered or degenerate is visible here
and nowhere else. Reports on REAL rows only (skips are query/probe copies).
"""
import os, sys, json, io

os.environ.setdefault("RWKV_ID_FEATURES", "1")
os.environ.setdefault("RWKV_REAL_CYCLES", "1")
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import lmdb
from rwkv.data_processing import CARD_FEATURE_COLUMNS
from rwkv.utils import load_tensor

DB = os.environ.get("AUDIT_DB", "F:/rwkv_lmdb/train_db_5k_h1_id5")
USERS = [int(x) for x in os.environ.get("AUDIT_USERS", "1,17,101,333,477,1503").split(",")]
COLS = list(CARD_FEATURE_COLUMNS)
print(f"db={DB}  columns={len(COLS)}")

env = lmdb.open(DB, readonly=True, lock=False, max_readers=4)
rows = []
skips_all = []
with env.begin() as txn:
    for u in USERS:
        raw = txn.get(f"{u}_batches".encode())
        if raw is None:
            print(f"user {u}: absent")
            continue
        keys = json.loads(raw)
        keys = sorted(keys)[:2]
        for (s, e, L) in keys:
            p = f"{u}_{s}-{e}_{L}_"
            cf = load_tensor(txn, p + "card_features", "cpu").float().numpy()
            sk = load_tensor(txn, p + "skips", "cpu").numpy().astype(bool)
            rows.append(cf)
            skips_all.append(sk)
        print(f"user {u}: {len(keys)} chunk(s), width {rows[-1].shape[1]}")
env.close()

X = np.concatenate(rows, 0)
S = np.concatenate(skips_all, 0)
assert X.shape[1] == len(COLS), f"stored width {X.shape[1]} != {len(COLS)} columns"
R = X[~S]
print(f"\nrows total {X.shape[0]}, real {R.shape[0]}, skip {int(S.sum())}\n")

print(f"{'#':>3} {'column':<28} {'mean':>9} {'std':>8} {'min':>9} {'max':>9} "
      f"{'%mode':>7} {'nuniq':>7}")
flags = []
for j, name in enumerate(COLS):
    v = R[:, j]
    vals, cnt = np.unique(v, return_counts=True)
    mode_v = vals[cnt.argmax()]
    mode_f = cnt.max() / v.size
    note = ""
    if v.std() == 0:
        note = "CONSTANT"
    elif mode_f > 0.5:
        note = f"mode={mode_v:+.4f}"
    print(f"{j:>3} {name:<28} {v.mean():>9.4f} {v.std():>8.4f} {v.min():>9.4f} "
          f"{v.max():>9.4f} {100*mode_f:>6.1f}% {vals.size:>7} {note}")
    if note:
        flags.append((name, mode_f, float(mode_v), vals.size))

print("\n--- columns whose single most common value covers >50% of real rows ---")
for name, f, mv, nu in sorted(flags, key=lambda t: -t[1]):
    print(f"  {name:<28} {100*f:5.1f}% at {mv:+.5f}  ({nu} distinct)")
