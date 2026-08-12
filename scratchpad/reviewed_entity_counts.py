#!/usr/bin/env python
"""Count REVIEWED entities per user across the whole anki-revlogs-10k dataset.

    cards reviewed at least once      = distinct card_id appearing in that user's revlogs
    notes with >=1 card reviewed once = distinct note_id whose card_id appears in the revlogs

These are the DEPLOY-relevant populations: RNN state exists only for an entity that has been
reviewed, never for a card sitting unstudied in a collection. Sizing off collection counts
overstates the worst case by ~50x, because the million-card users are imported shared decks --
user 629 has 1,256,705 cards and reviewed 2.0% of them.

Companion to scratchpad/entity_counts.py (which counts COLLECTION entities). Writes a per-user CSV
so percentiles can be recomputed without re-reading 745M reviews.

⚠ A card can be reviewed and later DELETED: it stays in revlogs but leaves the cards table. Such
card_ids are counted as reviewed (state would have existed for them) but contribute no note, since
there is no row to map them through. `reviewed_cards_unmapped` records how many, so the effect is
visible rather than silently folded into the note count.

Usage: python scratchpad/reviewed_entity_counts.py [out.csv] [--workers 6]
"""
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = "C:/Users/Andrew/anki-revlogs-10k"


def one_user(uid):
    cf = sorted(glob.glob(f"{ROOT}/cards/user_id={uid}/*.parquet"))
    rf = sorted(glob.glob(f"{ROOT}/revlogs/user_id={uid}/*.parquet"))
    if not cf or not rf:
        return None
    rev = pq.read_table(rf, columns=["card_id"]).column("card_id").combine_chunks()
    n_reviews = len(rev)
    reviewed = pc.unique(rev)
    n_rev_cards = len(reviewed)

    ct = pq.read_table(cf, columns=["card_id", "note_id"])
    n_coll_cards = ct.num_rows
    n_coll_notes = len(pc.unique(ct.column("note_id").combine_chunks()))
    # notes whose card was reviewed: filter the cards table to reviewed card_ids, then unique note_id
    mask = pc.is_in(ct.column("card_id"), value_set=reviewed)
    kept = ct.filter(mask)
    n_rev_notes = len(pc.unique(kept.column("note_id").combine_chunks())) if kept.num_rows else 0
    unmapped = n_rev_cards - kept.num_rows  # reviewed card_ids with no row in the cards table
    return (uid, n_coll_cards, n_rev_cards, n_coll_notes, n_rev_notes, n_reviews, unmapped)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else "scratchpad/reviewed_entity_counts_10k.csv"
    workers = 6
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    uids = sorted(int(os.path.basename(p).split("=")[1])
                  for p in glob.glob(f"{ROOT}/cards/user_id=*"))
    print(f"{len(uids)} users, {workers} workers -> {out}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one_user, uids, chunksize=8)):
            if r:
                rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  ...{i + 1}/{len(uids)}", flush=True)

    hdr = ["uid", "collection_cards", "reviewed_cards", "collection_notes",
           "reviewed_notes", "reviews", "reviewed_cards_unmapped"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} users)", flush=True)


if __name__ == "__main__":
    main()
