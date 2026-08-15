#!/usr/bin/env python
"""Do the deck ids STORED IN THE LMDB resolve against the parent map? CPU, ~1 min.

This is the one assumption the whole no-rebuild plan rests on, and until now it was spot-checked on
a SINGLE user (14/14, FUTURE_FEATURES.md, which itself says "widen that"). It is genuinely separable
from the parquet-side resolve rate: the parquet could be a perfect tree while the LMDB stored
something else (a factorized code, a merge artifact, the ID_PLACEHOLDER fill). Only reading the
stored ids answers it.

⚠ REVIEW-WEIGHTING IS THE RIGHT DENOMINATOR. Distinct-id weighting over-counts tiny decks: a user's
one enormous deck and their forty 3-card decks count the same. What bounds the lever is the fraction
of actual REVIEWS that can walk up at least one level.

Rows whose deck has no row in the decks table (deleted / filtered decks -- `df_decks` is merged
how="left", so a card may reference a deck with no row) are handled EXACTLY like roots: they bypass.
They are not a correctness problem, only a bound on reach.

ASCII output only -- this gets redirected into cp1252 logs.
"""
import argparse
import io
import json
import os
import sys

import lmdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_tensor(txn, key):
    raw = txn.get(key.encode())
    return None if raw is None else torch.load(io.BytesIO(raw), weights_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="train_db_5k_h1")
    ap.add_argument("--map", default="scratchpad/deck_tree/parent_maps.parquet")
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--all-chunks", action="store_true")
    a = ap.parse_args()

    from rwkv.data_processing import ID_PLACEHOLDER as IDP

    pm = pd.read_parquet(a.map)
    known, par = {}, {}
    for uid, g in pm.groupby("user_id"):
        known[int(uid)] = set(g["deck_id"].astype(int))
        par[int(uid)] = dict(zip(g["deck_id"].astype(int), g["parent_id"].astype(int)))

    env = lmdb.open(a.db, readonly=True, lock=False, subdir=True,
                    map_size=400_000_000_000, max_readers=2048)
    tot = unknown = placeholder = resolvable = 0
    r_tot = r_norow = r_root = r_anc = 0
    checked = 0
    with env.begin() as txn:
        for uid in sorted(known):
            if checked >= a.users:
                break
            raw = txn.get(f"{uid}_batches".encode())
            if raw is None:
                continue
            keys = json.loads(raw)
            if not keys:
                continue
            # SPREAD the sample across the user's history. Reading only keys[0] biases hard
            # toward a user's EARLIEST reviews, whose decks are the most likely to have been
            # deleted since -- which inflates the "no deck row" share.
            pick = keys if a.all_chunks else keys[:: max(1, len(keys) // 4)][:4]
            parts = []
            for s_, e_, L in pick:
                t = load_tensor(txn, f"{uid}_{s_}-{e_}_{L}_deck_id_id_")
                if t is not None:
                    parts.append(np.asarray(t).astype(np.int64))
            if not parts:
                continue
            arr = np.concatenate(parts)
            vals, counts = np.unique(arr, return_counts=True)
            for v, c in zip(vals, counts):
                v = int(v)
                tot += 1
                r_tot += int(c)
                if v == IDP:
                    placeholder += 1
                    r_norow += int(c)
                elif v not in known[uid]:
                    unknown += 1
                    r_norow += int(c)
                elif par[uid][v] != -1:
                    resolvable += 1
                    r_anc += int(c)
                else:
                    r_root += int(c)
            checked += 1

    pct = lambda x, n: f"{x/max(n,1):.2%}"
    print(f"checked {checked} users, {tot} distinct stored deck ids, {r_tot:,} reviews")
    print("  -- by DISTINCT id --")
    print(f"  no deck row        : {unknown} ({pct(unknown,tot)})")
    print(f"  ID_PLACEHOLDER     : {placeholder}")
    print(f"  has a real parent  : {resolvable} ({pct(resolvable,tot)})")
    print("  -- by REVIEW (bounds how much corpus the tree can touch) --")
    print(f"  no deck row (bypass like a root) : {r_norow:,} ({pct(r_norow,r_tot)})")
    print(f"  known root, no parent            : {r_root:,} ({pct(r_root,r_tot)})")
    print(f"  HAS an ancestor                  : {r_anc:,} ({pct(r_anc,r_tot)})")
    print("")
    print(f"VERDICT: {pct(r_anc,r_tot)} of reviews can walk up at least one level.")
    print("  Deck-row-less rows are treated exactly like roots (bypass), so they are not a")
    print("  correctness problem -- they only bound the lever's reach.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
