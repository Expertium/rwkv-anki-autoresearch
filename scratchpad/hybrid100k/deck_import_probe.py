"""Does an imported deck's deck_id carry the IMPORT date or the original author's date?

Andrew's question, and it decides whether a deck-age / card-age feature means what it says.
Anki ids ARE creation timestamps, so the question is what happens to them on .apkg import.
Rather than reason about Anki internals, measure it -- the -id dataset has the raw ids.

TWO INDEPENDENT TESTS, because either alone is ambiguous:

  1. CROSS-USER ID OVERLAP. A shared deck is downloaded by many users. If an id survives
     import, the SAME id appears in many users' collections; if it is reassigned at import
     time, it is unique per user (two users importing at different moments get different ids).
     The record already notes this happens for CARD ids -- scratchpad/dataset_id/ds20k/overlap.py
     found shared decks propagate card ids, which is why a card-id fingerprint reported 64%
     collection overlap instead of the true 4.3%. This checks all three id kinds the same way.

  2. CARD OLDER THAN ITS OWN DECK. You cannot add a card to a deck that does not exist yet, so
     for a natively-created card `card_id > deck_id` always. A card whose id PREDATES its deck's
     is therefore proof that the two were assigned by different events: the card kept an older
     original timestamp while the deck got a fresh one. That is the import signature, and it is
     per-row rather than statistical.

Read-only over ../anki-revlogs-10k-id.
"""
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

ROOT = r"C:/Users/Andrew/anki-revlogs-10k-id"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def load(table, uid, cols):
    d = os.path.join(ROOT, table, "user_id=%d" % uid)
    if not os.path.isdir(d):
        return None
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet")]
    if not fs:
        return None
    return pq.read_table(fs[0], columns=cols)


users = sorted(int(x.split("=")[1]) for x in os.listdir(os.path.join(ROOT, "cards")))
step = max(1, len(users) // N)
sample = users[::step][:N]

seen = {"card_id": Counter(), "note_id": Counter(), "deck_id": Counter()}
tot_cards = 0
older = 0
comparable = 0
per_user = []
n_ok = 0

for u in sample:
    c = load("cards", u, ["card_id", "note_id", "deck_id"])
    if c is None or c.num_rows == 0:
        continue
    n_ok += 1
    cid = c["card_id"].to_numpy()
    nid = c["note_id"].to_numpy()
    did = c["deck_id"].to_numpy()
    tot_cards += len(cid)
    for k, arr in (("card_id", cid), ("note_id", nid), ("deck_id", did)):
        seen[k].update(np.unique(arr).tolist())
    # test 2: a real Anki id is a ~1.1e12..1.8e12 epoch-ms value. The published set's tiny
    # factorized ids and Anki's reserved deck id 1 ("Default") are not timestamps; excluding
    # them is what keeps this test about imports rather than about sentinels.
    real = (did > 10**11) & (cid > 10**11)
    comparable += int(real.sum())
    o = int((cid[real] < did[real]).sum())
    older += o
    if real.sum():
        per_user.append((u, int(real.sum()), o, 100.0 * o / int(real.sum())))

print("sampled %d users with cards, %d cards\n" % (n_ok, tot_cards))
print("TEST 1 -- how often does the same id appear in MORE THAN ONE user's collection?")
print("   %-9s %10s %10s %8s" % ("id kind", "distinct", "shared", "share %"))
for k in ("card_id", "note_id", "deck_id"):
    tot = len(seen[k])
    sh = sum(1 for v in seen[k].values() if v > 1)
    print("   %-9s %10d %10d %7.2f%%" % (k, tot, sh, 100.0 * sh / tot if tot else 0))
print()
print("TEST 2 -- cards whose id PREDATES their own deck's id (impossible unless imported)")
print("   %d of %d comparable cards = %.2f%%" % (older, comparable,
                                                 100.0 * older / comparable if comparable else 0))
print()
print("   worst 8 users:")
for u, n, o, pct in sorted(per_user, key=lambda t: -t[3])[:8]:
    print("      user %-5d %6d cards, %6d older than their deck  %6.2f%%" % (u, n, o, pct))
frac = [p for _, _, _, p in per_user]
if frac:
    print("   per-user: mean %.2f%%  median %.2f%%  users with >10%%: %d of %d"
          % (np.mean(frac), np.median(frac), sum(1 for p in frac if p > 10), len(frac)))
