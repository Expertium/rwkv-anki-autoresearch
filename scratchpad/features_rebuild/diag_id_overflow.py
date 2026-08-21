"""Are the ids stored in the -id LMDBs TRUNCATED by int32?

MECHANISM. data_processing.create_sample stores every id stream as
    ids[submodule] = torch.tensor(section_df[submodule].to_numpy(), dtype=torch.int32)
The PUBLISHED set carries per-user FACTORIZED small ints, which fit. The -id set deliberately keeps
RAW ANKI EPOCH-MS ids (~1.7e12), which do NOT fit in int32 (max 2,147,483,647) and wrap silently.

WHY IT MATTERS FAR BEYOND THE CRASH. Ids are not just labels here:
  * insert_probes groups by id to find each card's first in-chunk row -- a collision makes one
    card's genuine FIRST review look like a repeat, so it becomes probe-eligible while having no
    query row. That is the KeyError that killed featB's fetch worker.
  * prepare_batch builds the per-entity ID ENCODING from `set(ids[submodule].tolist())` -- colliding
    cards share an encoding.
  * build_module_data GROUPS ROWS BY ID to form the recurrent streams -- a collision merges two
    cards' histories into ONE state. That is a correctness problem in the model, not just a crash.

Two cards collide iff their creation timestamps differ by an exact multiple of 2^32 ms (~49.71
days), so collisions are rare per pair but a birthday problem over tens of thousands of cards.

This reads the RAW parquet and the STORED tensors and compares.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/diag_id_overflow.py [n_users]
"""
import json
import os
import sys
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RWKV_ID_FEATURES", "0")
from rwkv.prepare_batch import get_data  # noqa: E402

IDD = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
INT32_MAX = 2 ** 31 - 1

print("--- 1. do the RAW -id ids exceed int32 at all?")
over = {}
for uid in (1, 2, 101, 209):
    c = pd.read_parquet(IDD / "cards", filters=[("user_id", "=", uid)])
    for col in ("card_id", "note_id", "deck_id"):
        if col not in c.columns:
            continue
        v = pd.to_numeric(c[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        v = v[v > 0]
        if v.size:
            frac = float((v > INT32_MAX).mean())
            over.setdefault(col, []).append(frac)
    print("  user %-4d cards=%-7d max card_id=%.0f  (int32 max = %d)"
          % (uid, len(c), pd.to_numeric(c["card_id"], errors="coerce").max(), INT32_MAX))
for col, fr in over.items():
    print("  %-8s fraction above int32 max: %s" % (col, ["%.3f" % f for f in fr]))

print("")
print("--- 2. what is actually STORED in the gen-2 lmdb?")
env = lmdb.open("F:/rwkv_lmdb/train_db_5k_h1_id2", map_size=400_000_000_000,
                readonly=True, lock=False)
with env.begin(write=False) as txn:
    raw = txn.get(b"1_batches")
    batch = json.loads(raw)[0]
    data = get_data(txn, (1, batch[0], batch[1], batch[2]), device="cpu")
    for name in ("card_id", "note_id", "deck_id"):
        t = data.ids[name]
        a = t.numpy()
        print("  %-8s dtype=%-8s min=%-14d max=%-14d n_unique=%d"
              % (name, t.dtype, a.min(), a.max(), len(np.unique(a))))
    neg = int((data.ids["card_id"].numpy() < 0).sum())
    print("  NEGATIVE card_id values (the unmistakable wrap signature): %d of %d"
          % (neg, data.ids["card_id"].numel()))

print("")
print("--- 3. does truncation actually COLLIDE distinct cards, per user?")
tot_users = int(sys.argv[1]) if len(sys.argv) > 1 else 40
stride = max(1, 5000 // tot_users)
hit = 0
checked = 0
worst = []
for uid in list(range(1, 5001, stride))[:tot_users]:
    p = IDD / "cards"
    try:
        c = pd.read_parquet(p, filters=[("user_id", "=", uid)])
    except Exception:  # noqa: BLE001
        continue
    v = pd.to_numeric(c["card_id"], errors="coerce").dropna().to_numpy(dtype=np.int64)
    if v.size < 2:
        continue
    checked += 1
    trunc = v.astype(np.int32).astype(np.int64)
    n_col = len(v) - len(np.unique(trunc))
    if n_col:
        hit += 1
        worst.append((uid, len(v), n_col))
print("  users checked: %d   users WITH at least one card_id collision: %d" % (checked, hit))
for uid, n, nc in sorted(worst, key=lambda x: -x[2])[:8]:
    print("    user %-5d %6d cards -> %d colliding" % (uid, n, nc))
print("")
if hit:
    print("CONFIRMED: int32 truncation collides distinct cards in %d/%d users." % (hit, checked))
    print("That is the KeyError mechanism, and it also merges two cards into one recurrent state.")
else:
    print("No collisions in this sample -- widen it before concluding the mechanism is absent.")
