"""REVIEW-weighted deck-tree depth: how many ancestor levels would an iterative
coarsening loop actually need? (Andrew 2026-07-24, arbitrary-depth tree question.)

Per review: depth of the card's deck (hops to root) and the number of DISTINCT
ancestors it has. Review-weighted, since that is what the compute cap must cover.
Also: how much does each level actually COARSEN (entities per level)?
"""
import glob
import os
from collections import Counter

import pandas as pd

PUB = r"C:\Users\Andrew\anki-revlogs-10k"
users = sorted(glob.glob(os.path.join(PUB, "decks", "user_id=*")),
               key=lambda p: int(p.rsplit("=", 1)[1]))
sample = [users[i] for i in range(0, len(users), max(1, len(users) // 80))][:80]

depth_hist = Counter()
entities_per_level = Counter()   # level -> total distinct ancestor entities
rows_per_level = Counter()       # level -> reviews that HAVE an ancestor there
tot_rev = 0
for u in sample:
    uid = u.rsplit("=", 1)[1]
    cpath = os.path.join(PUB, "cards", f"user_id={uid}", "data.parquet")
    rpath = os.path.join(PUB, "revlogs", f"user_id={uid}", "data.parquet")
    if not (os.path.exists(cpath) and os.path.exists(rpath)):
        continue
    decks = pd.read_parquet(os.path.join(u, "data.parquet"))
    cards = pd.read_parquet(cpath)
    rev = pd.read_parquet(rpath, columns=["card_id"])
    par = dict(zip(decks["deck_id"], decks["parent_id"]))
    ids = set(decks["deck_id"])

    chain_cache = {}

    def chain(did):
        if did in chain_cache:
            return chain_cache[did]
        out, cur, seen = [], did, set()
        while True:
            p = par.get(cur)
            if p is None or p not in ids or p in seen:
                break
            seen.add(p)
            out.append(p)
            cur = p
        chain_cache[did] = out
        return out

    deck_of = dict(zip(cards["card_id"], cards["deck_id"]))
    per_level_ents = {}
    for cid in rev["card_id"]:
        did = deck_of.get(cid)
        if did is None:
            continue
        ch = chain(did)
        tot_rev += 1
        depth_hist[len(ch)] += 1
        for lvl, anc in enumerate(ch, start=1):
            rows_per_level[lvl] += 1
            per_level_ents.setdefault(lvl, set()).add(anc)
    for lvl, s in per_level_ents.items():
        entities_per_level[lvl] += len(s)

print(f"{tot_rev:,} reviews over {len(sample)} sampled users")
print("\nreview-weighted DEPTH of the card's own deck (hops to root):")
cum = 0
for d in sorted(depth_hist):
    cum += depth_hist[d]
    print(f"  depth {d}: {depth_hist[d]:>9,}  ({100*depth_hist[d]/tot_rev:5.1f}%)  "
          f"cumulative {100*cum/tot_rev:5.1f}%")
print("\nreviews that HAVE an ancestor at level L (= work an L-level loop does):")
for lvl in sorted(rows_per_level):
    print(f"  level {lvl}: {rows_per_level[lvl]:>9,} reviews "
          f"({100*rows_per_level[lvl]/tot_rev:5.1f}% of all), "
          f"{entities_per_level[lvl]:>6,} distinct ancestor entities")
