#!/usr/bin/env python
"""How DEEP does the deck tree actually go, review-weighted? Chooses L for RWKV_DECK_TREE=L.

verify_lmdb_link.py answered "can a row walk up AT LEAST one level" (49.21%). That is the reach of
L=2. Each further level costs a full extra RWKV stream in the chain -- the same per-step cost as the
deck stream itself -- so the question that sizes the lever is the MARGINAL reach of level 3, 4, ...

Counts, per level L, the reviews whose deck has an ancestor at distance L. A row that runs out of
ancestors bypasses from there on, so reach is monotonically non-increasing in L.

⚠ Sample ALL chunks (the keys[0] trap: a user's earliest decks are the most likely to have been
deleted, which inflates the deck-row-less share and deflates reach). ASCII output only.
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
    ap.add_argument("--max-level", type=int, default=6)
    a = ap.parse_args()

    from rwkv.data_processing import ID_PLACEHOLDER as IDP

    pm = pd.read_parquet(a.map)
    par = {}
    for uid, g in pm.groupby("user_id"):
        par[int(uid)] = dict(zip(g["deck_id"].astype(int), g["parent_id"].astype(int)))

    env = lmdb.open(a.db, readonly=True, lock=False, subdir=True,
                    map_size=400_000_000_000, max_readers=2048)
    L = a.max_level
    r_tot = 0
    reach = np.zeros(L + 1, dtype=np.int64)   # reach[k] = reviews with an ancestor at distance k
    # per-deck depth histogram, review-weighted, to show where the mass sits
    depth_hist = {}
    checked = 0
    with env.begin() as txn:
        for uid in sorted(par):
            if checked >= a.users:
                break
            raw = txn.get(f"{uid}_batches".encode())
            if raw is None:
                continue
            keys = json.loads(raw)
            if not keys:
                continue
            parts = []
            for s_, e_, Ln in keys:
                t = load_tensor(txn, f"{uid}_{s_}-{e_}_{Ln}_deck_id_id_")
                if t is not None:
                    parts.append(np.asarray(t).astype(np.int64))
            if not parts:
                continue
            arr = np.concatenate(parts)
            vals, counts = np.unique(arr, return_counts=True)
            p = par[uid]
            for v, c in zip(vals, counts):
                v = int(v)
                c = int(c)
                r_tot += c
                if v == IDP or v not in p:
                    depth_hist[0] = depth_hist.get(0, 0) + c
                    continue
                # walk up
                cur = v
                d = 0
                seen = {cur}
                while d < L:
                    nxt = p.get(cur, -1)
                    if nxt == -1 or nxt in seen:
                        break
                    d += 1
                    reach[d] += c
                    seen.add(nxt)
                    cur = nxt
                depth_hist[d] = depth_hist.get(d, 0) + c
            checked += 1

    pct = lambda x: f"{x/max(r_tot,1):.2%}"
    print(f"checked {checked} users, {r_tot:,} reviews")
    print("")
    print("  REACH by level (reviews whose deck has an ancestor at that distance):")
    for k in range(1, L + 1):
        marg = reach[k] - (reach[k + 1] if k + 1 <= L else 0)
        print(f"    level {k+1} (ancestor at distance {k}) : {reach[k]:>12,}  {pct(reach[k]):>8}"
              f"   (deepest here: {pct(marg)})")
    print("")
    print("  ANCESTOR-CHAIN DEPTH histogram, review-weighted:")
    for d in sorted(depth_hist):
        print(f"    depth {d}: {depth_hist[d]:>12,}  {pct(depth_hist[d]):>8}")
    print("")
    print("  L=2 covers the level-1 row; L=3 adds level-2; etc. Marginal reach is what an extra")
    print("  stream in the chain buys -- compare it against a full extra RWKV stream's cost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
