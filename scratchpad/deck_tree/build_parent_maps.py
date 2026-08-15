#!/usr/bin/env python
"""Build the per-user deck_id -> parent_id map that the deck-TREE streams need. CPU, minutes.

WHY NO LMDB REBUILD IS NEEDED (the load-bearing fact, confirmed structurally here rather than by
spot check). `data_processing.get_rwkv_data` stores `deck_id` RAW from the user's decks parquet --
it drops `parent_id` (:228) but never factorizes or remaps `deck_id`; the only rewrite is NaN ->
ID_PLACEHOLDER (:243-245). So the parquet's own deck_id -> parent_id mapping applies DIRECTLY to
the ids already inside the LMDBs, and ancestor groupings can be built at batch time from the same
~20-line grouping code `prepare_batch.insert_probes` already uses. The alternative -- threading
parent_id through preprocessing -- costs a 2-4 day rebuild of a 372.5 GB database that does not
fit side by side on C:. FUTURE_FEATURES.md's older design sketch still says "needs LMDB rebuild";
that half is superseded by its own 2026-07-26 correction.

⚠ THE 5.5% THAT DO NOT RESOLVE ARE NOT ERRORS. Top-level decks carry a `0` root sentinel that was
factorized into a per-user code which is not itself a deck row. That is "no parent", and those
rows must BYPASS at every level rather than being grouped together -- pooling all roots into one
pseudo-deck would invent a relationship the data does not have.

Output: scratchpad/deck_tree/parent_maps.parquet with columns (user_id, deck_id, parent_id),
parent_id = -1 where the deck has no resolvable parent. Small: ~56 decks for a median user.

Usage: python scratchpad/deck_tree/build_parent_maps.py [--users 1 7500] [--out <path>]
"""
import argparse, os, sys
from collections import Counter
import numpy as np
import pandas as pd

DATA = r"C:\Users\Andrew\anki-revlogs-10k"      # READ-ONLY
NO_PARENT = -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs=2, type=int, default=[1, 7500])
    ap.add_argument("--out", default="scratchpad/deck_tree/parent_maps.parquet")
    a = ap.parse_args()
    lo, hi = a.users

    rows, missing, stats = [], 0, Counter()
    depth_hist = Counter()
    for uid in range(lo, hi + 1):
        d = os.path.join(DATA, "decks", f"user_id={uid}")
        if not os.path.isdir(d):
            missing += 1
            continue
        df = pd.read_parquet(d, columns=["deck_id", "parent_id"])
        ids = set(df["deck_id"].tolist())
        par = {int(k): (int(v) if int(v) in ids else NO_PARENT)
               for k, v in zip(df["deck_id"], df["parent_id"])}
        # self-parent would make the ancestor walk loop forever; treat as root and count it
        for k in list(par):
            if par[k] == k:
                par[k] = NO_PARENT
                stats["self_parent"] += 1
        stats["decks"] += len(par)
        stats["resolved"] += sum(1 for v in par.values() if v != NO_PARENT)
        # depth via the walk we will actually use at batch time (with a cycle guard)
        for k in par:
            seen, cur, dep = {k}, k, 0
            while par.get(cur, NO_PARENT) != NO_PARENT and dep < 32:
                cur = par[cur]
                if cur in seen:
                    stats["cycle"] += 1
                    break
                seen.add(cur); dep += 1
            depth_hist[dep] += 1
        rows.extend((uid, k, v) for k, v in par.items())
        if uid % 1000 == 0:
            print(f"  ...user {uid}: {len(rows):,} deck rows so far", flush=True)

    out = pd.DataFrame(rows, columns=["user_id", "deck_id", "parent_id"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_parquet(a.out, index=False)

    r = stats["resolved"] / max(stats["decks"], 1)
    print(f"\nusers {lo}-{hi}: {len(out):,} deck rows, {missing} users with no decks partition")
    print(f"  resolve rate {r:.3%}  (expect ~94.5%; the rest are top-level 'no parent')")
    print(f"  self-parents {stats['self_parent']}   cycles {stats['cycle']}")
    print("  deck-weighted depth histogram:",
          {k: depth_hist[k] for k in sorted(depth_hist)[:9]})
    print(f"  -> {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
    if stats["cycle"]:
        print("  ⚠ CYCLES FOUND -- the batch-time ancestor walk must keep its depth guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
