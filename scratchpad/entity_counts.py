"""Loop over the anki-revlogs-10k dataset and report per-user entity counts (cards, notes, decks,
presets) -- the entity counts that drive deploy state memory (state cost ~ entity count per stream).

Definitions (state-relevant = entities the model actually builds state for, i.e. that contain cards):
  cards   = rows in cards/user_id=N (one row per card)
  notes   = distinct note_id in cards
  decks   = distinct deck_id in cards (decks that contain >=1 card)
  presets = distinct preset_id among those decks (join cards.deck_id -> decks.preset_id)
READ-ONLY: only reads the dataset, never writes to it.
"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

BASE = r"C:\Users\Andrew\anki-revlogs-10k"


def count_user(uid):
    try:
        cards = pd.read_parquet(fr"{BASE}\cards\user_id={uid}\data.parquet",
                                columns=["card_id", "note_id", "deck_id"])
    except Exception:
        return None
    n_cards = len(cards)
    n_notes = int(cards["note_id"].nunique())
    decks_with_cards = set(cards["deck_id"].unique().tolist())
    n_decks = len(decks_with_cards)
    n_presets = -1
    try:
        decks = pd.read_parquet(fr"{BASE}\decks\user_id={uid}\data.parquet",
                                columns=["deck_id", "preset_id"])
        d2p = dict(zip(decks["deck_id"].tolist(), decks["preset_id"].tolist()))
        presets = {d2p[d] for d in decks_with_cards if d in d2p}
        n_presets = len(presets)
    except Exception:
        pass
    return (uid, n_cards, n_notes, n_decks, n_presets)


def main():
    uids = sorted(int(d.split("=")[1]) for d in os.listdir(fr"{BASE}\cards")
                  if d.startswith("user_id="))
    print(f"users found: {len(uids)}")
    rows = []
    with Pool(7) as pool:
        for i, r in enumerate(pool.imap_unordered(count_user, uids, chunksize=20), 1):
            if r is not None:
                rows.append(r)
            if i % 1000 == 0:
                print(f"  processed {i}/{len(uids)}")
    arr = pd.DataFrame(rows, columns=["uid", "cards", "notes", "decks", "presets"])
    arr.to_csv(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\entity_counts_10k.csv", index=False)
    print(f"\nusers with valid data: {len(arr)}  (preset failures: {(arr['presets']<0).sum()})")
    print(f"{'metric':>8} | {'mean':>10} | {'median':>8} | {'p90':>8} | {'p99':>8} | {'max':>9} | {'min':>5} | {'total':>14}")
    print("-" * 92)
    for col in ["cards", "notes", "decks", "presets"]:
        s = arr[col][arr[col] >= 0] if col == "presets" else arr[col]
        print(f"{col:>8} | {s.mean():>10.1f} | {int(s.median()):>8} | {int(s.quantile(.90)):>8} | "
              f"{int(s.quantile(.99)):>8} | {int(s.max()):>9} | {int(s.min()):>5} | {int(s.sum()):>14,}")
    # ratios useful for deploy memory weighting
    print(f"\nnotes/cards ratio (median): {(arr['notes']/arr['cards'].clip(lower=1)).median():.3f}")
    print(f"cards/deck (median): {(arr['cards']/arr['decks'].clip(lower=1)).median():.1f}")
    print(f"cards/preset (median): {(arr['cards']/arr['presets'].clip(lower=1)).median():.1f}")


if __name__ == "__main__":
    main()
