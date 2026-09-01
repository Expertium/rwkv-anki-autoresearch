"""Are FILTERED (cram) decks present in the -id dataset, and do cards point at them?

WHY IT MATTERS. A filtered deck is created fresh, holds cards temporarily, and is deleted.
If `cards.deck_id` carries the filtered deck (Anki's `did`) rather than the home deck
(`odid`), then `max(card_id, deck_id)` -- the acquisition-time proposal in
FUTURE_FEATURES.md -- would read "acquired yesterday" for a card owned for years. That is a
far worse error than the 13.6% "moved" case, so it has to be checked, not assumed.

The proto gives us no `odid`: CardEntry is (id, note_id, deck_id) only. So we cannot read the
flag directly and must look for the SYMPTOMS:

  1. A deck id NEWER than the card's LAST review. A home deck exists while the card is being
     studied. A deck created after the card's final review cannot have been the home deck at
     the time -- it is either a filtered deck captured at export, or a late reorganization.
  2. Preset sentinels. Anki's filtered decks carry no normal preset, so an unusual preset_id
     (0, or a value shared by nothing else) would mark them.
  3. Tiny, very recent decks. Filtered decks are transient and hold a day's worth of cards.

Read-only over ../anki-revlogs-10k-id.
"""
import os
import sys
from collections import Counter

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

preset_vals = Counter()
parent_vals = Counter()
tot = after_last = 0
deck_sizes = []
n_decks = 0
for u in sample:
    c = load("cards", u, ["card_id", "deck_id"])
    r = load("revlogs", u, ["card_id", "review_time"])
    d = load("decks", u, ["deck_id", "parent_id", "preset_id"])
    if c is None or r is None or c.num_rows == 0:
        continue
    if d is not None:
        n_decks += d.num_rows
        preset_vals.update(d["preset_id"].to_pylist())
        parent_vals.update(("zero" if p == 0 else "nonzero") for p in d["parent_id"].to_pylist())
    cid = c["card_id"].to_numpy()
    did = c["deck_id"].to_numpy()
    rc = r["card_id"].to_numpy()
    rt = r["review_time"].to_numpy()
    order = np.argsort(rc, kind="stable")
    rc, rt = rc[order], rt[order]
    starts = np.concatenate(([0], np.nonzero(np.diff(rc))[0] + 1))
    last_rt = dict(zip(rc[starts].tolist(), np.maximum.reduceat(rt, starts).tolist()))
    real = did > 10**11
    for c_, d_ in zip(cid[real].tolist(), did[real].tolist()):
        lr = last_rt.get(c_)
        if lr is None:
            continue
        tot += 1
        if d_ > lr:
            after_last += 1
    vals, cnt = np.unique(did, return_counts=True)
    deck_sizes.extend(cnt.tolist())

print("%d users, %d decks, %d cards with reviews\n" % (N, n_decks, tot))
print("TEST 1 -- deck_id NEWER than the card's LAST review (impossible for a live home deck)")
print("   %d of %d = %.3f%%\n" % (after_last, tot, 100.0 * after_last / tot if tot else 0))
print("TEST 2 -- preset_id values across all decks (a filtered deck would need an odd one)")
top = preset_vals.most_common(8)
print("   distinct preset ids: %d   most common: %s" % (len(preset_vals),
      ", ".join("%s x%d" % ("default(1)" if k == 1 else str(k), v) for k, v in top[:5])))
print("   any preset_id == 0? %s" % ("YES -- %d decks" % preset_vals.get(0, 0)
                                     if preset_vals.get(0) else "no"))
print("   parent_id: %s\n" % dict(parent_vals))
ds = np.array(deck_sizes)
print("TEST 3 -- deck size distribution (filtered decks are small and transient)")
print("   decks with 1 card: %d (%.1f%%)   median size %.0f   p90 %.0f"
      % (int((ds == 1).sum()), 100.0 * (ds == 1).mean(), np.median(ds), np.percentile(ds, 90)))
