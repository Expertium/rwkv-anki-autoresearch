"""How big is the SYNTHETIC deck that every deleted card is pooled into?

This is the real mechanism behind Andrew's proposal, and it is stronger than "extra data".
`data_processing.py` fills a missing deck_id/preset_id with a BARE CONSTANT (`ID_PLACEHOLDER`), so
all of a user's deleted cards land in ONE deck and ONE preset. (note_id gets `ID_PLACEHOLDER +
card_id`, which is unique per card, so notes are NOT pooled -- only deck and preset are.)

The deck stream is a per-deck RWKV recurrence: it pools evidence over the cards of a deck. If the
synthetic deck is the LARGEST deck in the user, then the deck stream spends most of its capacity
summarising a group whose members share nothing except having been deleted. That is not extra data,
it is a FABRICATED grouping, and it is a concrete reason to expect a gain rather than a null.

Prints the synthetic deck's rank and share against the user's real decks.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("../anki-revlogs-10k")
USERS = [3, 17, 101, 333, 555, 1200, 2500, 4800]


def main():
    print(f"  {'user':>6} {'real decks':>11} {'fake deck rank':>15} {'fake deck share':>17} "
          f"{'largest real deck':>19}")
    ranks = []
    for user_id in USERS:
        try:
            rl = pd.read_parquet(DATA / "revlogs" / f"{user_id=}", columns=["card_id"])
            cards = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", user_id)])
        except Exception as e:
            print(f"  {user_id:>6}  (skipped: {type(e).__name__})")
            continue
        m = cards.set_index("card_id")["deck_id"]
        deck = rl["card_id"].map(m)                    # NaN  <=>  the card was deleted
        n_fake = int(deck.isna().sum())
        real = deck.dropna().value_counts()
        if n_fake == 0 or real.empty:
            print(f"  {user_id:>6} {len(real):>11,}   (no deleted cards)")
            continue
        # where would the synthetic deck rank among the real ones, by review count?
        rank = int((real.to_numpy() > n_fake).sum()) + 1
        ranks.append(rank)
        tot = len(deck)
        print(f"  {user_id:>6} {len(real):>11,} {rank:>15,} {100*n_fake/tot:>16.2f}% "
              f"{real.max():>19,}")
    if ranks:
        print(f"\n  synthetic deck ranks #1 (the LARGEST deck in the user) for "
              f"{sum(1 for r in ranks if r == 1)}/{len(ranks)} users; median rank {int(np.median(ranks))}")


if __name__ == "__main__":
    main()
