"""Screen for Andrew's DELETED-CARDS proposal (2026-08-18): how much data is it, and is it SCORED?

THE PROPOSAL: reviews of cards the user has since deleted are still in the training set. Drop them
and see whether logloss degrades. If not, it is a free win.

PREMISE CHECK FIRST -- and it is right in mechanism, wrong in the sentinel. `data_processing.py`
merges revlogs -> cards -> decks with `how="left"`, so a deleted card yields NaN, and the NaN is
then filled with `ID_PLACEHOLDER = 314159265358979323` (lines 285-296), NOT -1. The distinction that
matters:

    note_id    <- ID_PLACEHOLDER + card_id   -> UNIQUE per card, so notes are not pooled
    deck_id    <- ID_PLACEHOLDER             -> a BARE CONSTANT
    preset_id  <- ID_PLACEHOLDER             -> a BARE CONSTANT

So every deleted card in a user collapses into ONE fake deck and ONE fake preset. That is the real
mechanism to worry about: the deck and preset RWKV streams pool those reviews into a single
synthetic entity whose members have nothing in common except having been deleted. It is not merely
extra data, it is a fabricated grouping.

THE TWO QUESTIONS THIS ANSWERS, both from the published dataset, no GPU:
  1. PREVALENCE -- what fraction of reviews and of cards are affected? An effect on 0.1% of rows
     cannot clear a 1e-4 gate; an effect on 5% can.
  2. ★ ARE THEY SCORED? This is the one that decides whether the experiment is even interpretable.
     If deleted cards appear in the EVAL set, then dropping them from training creates a
     train/test mismatch, and a degradation would not mean "the data was useful" -- it would mean
     "we stopped training on something we are still graded on". Those are different conclusions and
     the run cannot separate them.

Train range is 1-5000, eval range 5001-7500 (the VAL half).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../anki-revlogs-10k")
TRAIN = [1, 2, 3, 17, 101, 333, 555, 1200, 2500, 4800]
EVAL = [5001, 5050, 5200, 5600, 6104, 6500, 7000, 7499]


def scan(user_id):
    # NOTE the parameter name: the directories are `user_id=N`, and this repo's probes rely on
    # Python's self-documenting f-string (f"{user_id=}") to build that. Calling it `uid` silently
    # produces `uid=N` and every read raises FileNotFoundError.
    rl = pd.read_parquet(DATA / "revlogs" / f"{user_id=}", columns=["card_id"])
    cards = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", user_id)])
    known = set(cards["card_id"].tolist())
    all_ids = rl["card_id"].to_numpy()
    miss = ~np.isin(all_ids, list(known)) if known else np.ones(all_ids.size, bool)
    n_cards = len(np.unique(all_ids))
    n_missing_cards = len(np.unique(all_ids[miss])) if miss.any() else 0
    return rl.shape[0], int(miss.sum()), n_cards, n_missing_cards


def report(tag, users):
    print(f"\n=== {tag} ===")
    print(f"  {'user':>6} {'reviews':>10} {'deleted-card reviews':>22} {'cards':>8} {'deleted cards':>15}")
    tr = td = tc = tdc = 0
    for u in users:
        try:
            r, d, c, dc = scan(u)
        except Exception as e:
            print(f"  {u:>6}  (skipped: {type(e).__name__})")
            continue
        tr += r; td += d; tc += c; tdc += dc
        print(f"  {u:>6} {r:>10,} {d:>13,} ({100*d/max(r,1):5.2f}%) {c:>8,} {dc:>8,} ({100*dc/max(c,1):5.2f}%)")
    print(f"  {'TOTAL':>6} {tr:>10,} {td:>13,} ({100*td/max(tr,1):5.2f}%) {tc:>8,} {tdc:>8,} ({100*tdc/max(tc,1):5.2f}%)")
    return tr, td


if __name__ == "__main__":
    pass  # docstring holds the method; printing it trips cp1252 on the star glyph
    a = report("TRAIN range (1-5000) -- what would be dropped", TRAIN)
    b = report("EVAL range (5001-7500) -- what is still SCORED", EVAL)
    print("\n=== VERDICT INPUTS ===")
    print(f"  training rows that would be dropped : {100*a[1]/max(a[0],1):.2f}%")
    print(f"  eval rows that are deleted-card rows: {100*b[1]/max(b[0],1):.2f}%")
    print("  -> if the EVAL figure is NOT ~0, the experiment is confounded: dropping them from")
    print("     training while still being graded on them conflates 'the data was useful' with")
    print("     'we stopped training on what we are tested on'.")
