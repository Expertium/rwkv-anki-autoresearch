"""The CONTROL for the int32 id-overflow finding: is the OLD db intact?

The gen-2 db stores card_id with 7,693 of 15,191 values NEGATIVE, note_id with n_unique == 1, and
deck_id with n_unique == 2. If the OLD db shows healthy, distinct, positive ids on the same users,
the damage is specific to the -id rebuild and every published-set result stands. If the OLD db shows
it too, the blast radius is the entire project.

Also separates the two failure modes, which have different signatures and probably different causes:
  * card_id is int64 in the frame -> int32 conversion WRAPS -> a spread of values, some negative,
    with rare collisions;
  * note_id/deck_id/preset_id go through a NaN-fill that makes them float64 -> float->int32
    conversion SATURATES -> every value pinned at INT32_MIN, i.e. ONE entity.
Saturation is the catastrophic one: it destroys per-entity identity entirely, so the note stream
pools a whole user into a single recurrent state.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/diag_id_control.py
"""
import json
import os
import sys

import lmdb
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RWKV_ID_FEATURES", "0")
from rwkv.prepare_batch import get_data  # noqa: E402

DBS = [
    ("OLD  (published)", "train_db_5k_h1", 400_000_000_000),
    ("GEN2 (-id)", "F:/rwkv_lmdb/train_db_5k_h1_id2", 400_000_000_000),
    ("GEN1 (-id)", "F:/rwkv_lmdb/train_db_5k_h1_id", 400_000_000_000),
]
USERS = [1, 101, 209, 417]

for label, path, size in DBS:
    if not os.path.exists(path):
        print("%s: MISSING (%s)" % (label, path))
        continue
    print("=== %s" % label)
    try:
        env = lmdb.open(path, map_size=size, readonly=True, lock=False)
    except Exception as exc:  # noqa: BLE001
        print("   cannot open: %s" % exc)
        continue
    with env.begin(write=False) as txn:
        for uid in USERS:
            raw = txn.get(("%d_batches" % uid).encode())
            if raw is None:
                continue
            batch = json.loads(raw)[0]
            try:
                d = get_data(txn, (uid, batch[0], batch[1], batch[2]), device="cpu")
            except Exception as exc:  # noqa: BLE001
                print("   user %d unreadable: %s" % (uid, exc))
                continue
            bits = []
            for name in ("card_id", "note_id", "deck_id"):
                a = d.ids[name].numpy()
                bits.append("%s uniq=%d neg=%d" % (name.split("_")[0], len(np.unique(a)),
                                                   int((a < 0).sum())))
            print("   user %-5d rows=%-6d %s" % (uid, d.ids["card_id"].numel(), "  ".join(bits)))
    print("")
print("READ: healthy = many uniques, zero negatives. uniq==1 means the whole stream collapsed to a")
print("single entity, which removes the per-entity recurrent state the hierarchy is built on.")
