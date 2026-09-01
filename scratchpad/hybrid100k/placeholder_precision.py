"""Does `ID_PLACEHOLDER + card_id` survive the float64 column it is written into?

THE SHAPE. data_processing.py:286 fills a NaN note_id with `ID_PLACEHOLDER + card_id`,
"precisely so each such card gets a UNIQUE placeholder" (its own comment). But `df["note_id"]`
CONTAINS NaN at that moment, so pandas holds it as float64, and the assignment is done in
float64. ID_PLACEHOLDER is 314159265358979323 ~ 3.14e17, which is far beyond float64's exact
integer range of 2^53 ~ 9.0e15. The low bits of card_id are therefore rounded away BEFORE the
int64 cast in create_sample ever runs -- so the 2026-08-21 int32->int64 fix widened the
DESTINATION while the VALUE had already been destroyed upstream.

This is Bug A's shape in a different numeric regime, and it fails the same way: silently, with
distinct cards collapsing into one note entity, and no assert anywhere.

Measured on both id regimes, because they fail differently:
  * PUBLISHED: card_id is factorized to small ints, so every card in a 64-wide block collides.
  * -id:       card_id is a raw epoch-ms timestamp, so cards created within one ulp collide --
               which is exactly what a bulk add or an import produces.

Read-only. CPU, seconds.
"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from rwkv.data_processing import ID_PLACEHOLDER  # noqa: E402

print("ID_PLACEHOLDER = %d  (~%.3e)" % (ID_PLACEHOLDER, ID_PLACEHOLDER))
ulp = np.spacing(np.float64(ID_PLACEHOLDER))
print("float64 spacing at that magnitude: %.0f" % ulp)
print("=> any two card_ids closer than %.0f collapse to the SAME placeholder\n" % ulp)


def collisions(card_ids, label):
    a = np.asarray(card_ids, dtype=np.int64)
    a = np.unique(a)
    exact = (np.int64(ID_PLACEHOLDER) + a)                       # what was intended
    viafloat = (np.float64(ID_PLACEHOLDER) + a.astype(np.float64)).astype(np.int64)
    n_exact = len(np.unique(exact))
    n_float = len(np.unique(viafloat))
    print("  %-34s cards %7d   distinct placeholders: exact %7d   via float64 %7d   %s"
          % (label, len(a), n_exact, n_float,
             "OK" if n_float == n_exact else "*** %d LOST (%.1f%%)"
             % (n_exact - n_float, 100.0 * (n_exact - n_float) / n_exact)))
    return n_exact, n_float


for root, label in ((r"C:/Users/Andrew/anki-revlogs-10k", "PUBLISHED (factorized ids)"),
                    (r"C:/Users/Andrew/anki-revlogs-10k-id", "-id (raw epoch-ms ids)")):
    print(label)
    users = sorted(int(x.split("=")[1]) for x in os.listdir(os.path.join(root, "cards")))
    tot_e = tot_f = 0
    for u in users[::max(1, len(users) // 6)][:6]:
        d = os.path.join(root, "cards", "user_id=%d" % u)
        fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet")]
        if not fs:
            continue
        cid = pq.read_table(fs[0], columns=["card_id"])["card_id"].to_numpy()
        e, f = collisions(cid, "user %d" % u)
        tot_e += e
        tot_f += f
    print("  TOTAL distinct: exact %d  via float64 %d  -> %.1f%% of the intended identity lost\n"
          % (tot_e, tot_f, 100.0 * (tot_e - tot_f) / tot_e if tot_e else 0))
