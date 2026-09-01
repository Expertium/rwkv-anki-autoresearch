"""Split "card older than its deck" into IMPORTED vs MOVED -- they are not the same thing.

deck_import_probe.py found 77% of cards have card_id < deck_id. That is NOT 77% imported: a
user who studies in Default, later creates "Japanese" and moves the cards in produces exactly
the same signature with no import at all.

THE DISCRIMINATOR IS THE FIRST REVIEW.
  * MOVED  -- the card existed and was being STUDIED before the deck was created, so its first
              review PREDATES deck_id.
  * IMPORT -- you cannot review a card before you have it, so the first review comes AFTER
              deck_id even though card_id is older. A large (deck_id - card_id) gap with all
              reviews after deck_id is the import signature.

Read-only over ../anki-revlogs-10k-id.
"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = r"C:/Users/Andrew/anki-revlogs-10k-id"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def load(table, uid, cols):
    d = os.path.join(ROOT, table, "user_id=%d" % uid)
    if not os.path.isdir(d):
        return None
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet")]
    return pq.read_table(fs[0], columns=cols) if fs else None


users = sorted(int(x.split("=")[1]) for x in os.listdir(os.path.join(ROOT, "cards")))
step = max(1, len(users) // N)
sample = users[::step][:N]

tot = moved = imported = native = 0
gaps = []
for u in sample:
    c = load("cards", u, ["card_id", "deck_id"])
    r = load("revlogs", u, ["card_id", "review_time"])
    if c is None or r is None or c.num_rows == 0 or r.num_rows == 0:
        continue
    cid = c["card_id"].to_numpy()
    did = c["deck_id"].to_numpy()
    rc = r["card_id"].to_numpy()
    rt = r["review_time"].to_numpy()
    order = np.argsort(rc, kind="stable")
    rc, rt = rc[order], rt[order]
    starts = np.concatenate(([0], np.nonzero(np.diff(rc))[0] + 1))
    first_ids = rc[starts]
    first_rt = np.minimum.reduceat(rt, starts)
    lut = dict(zip(first_ids.tolist(), first_rt.tolist()))

    real = (did > 10**11) & (cid > 10**11)
    for c_, d_ in zip(cid[real].tolist(), did[real].tolist()):
        fr = lut.get(c_)
        if fr is None:
            continue
        tot += 1
        if c_ >= d_:
            native += 1
        elif fr < d_:
            moved += 1            # studied before the deck existed
        else:
            imported += 1         # older card, but never reviewed until after the deck appeared
            gaps.append((d_ - c_) / 86400000.0)

print("cards with a first review, real ids, %d users: %d\n" % (N, tot))
for name, n in (("native (card newer than its deck)", native),
                ("MOVED  (card studied before the deck existed)", moved),
                ("IMPORT-consistent (older card, first review after the deck)", imported)):
    print("  %-58s %8d  %5.1f%%" % (name, n, 100.0 * n / tot if tot else 0))
if gaps:
    g = np.array(gaps)
    print("\n  import-consistent card-to-deck gap, days: median %.0f  p90 %.0f  max %.0f"
          % (np.median(g), np.percentile(g, 90), g.max()))
    print("  (a genuine import shows a LARGE gap: the author made the card long before you got it)")
