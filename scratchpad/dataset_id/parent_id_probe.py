"""What is DeckEntry.parent_id in anki-revlogs-10k-id? (Andrew 2026-07-24)

Hypothesis: it is the deck_id of the parent deck (Anki decks are a "::"-separated
tree; each deck row could point at its parent). Tests across a user sample:
  1. coverage: how many deck rows have parent_id != 0
  2. closure: do non-zero parent_ids exist as deck_id in the SAME user's table?
  3. acyclicity + depth: does following parent_id terminate (a forest)?
  4. root convention: what value do top-level decks use (0? 1? self?)
  5. id semantics: are parent ids epoch-ms like deck ids (creation time), and is a
     parent ALWAYS older than its child (a tree built top-down would imply that)?
  6. preset correlation: do children share their parent's preset?
"""
import glob
import os
from collections import Counter

import pandas as pd

ROOT = r"C:\Users\Andrew\anki-revlogs-10k-id"
users = sorted(glob.glob(os.path.join(ROOT, "decks", "user_id=*")),
               key=lambda p: int(p.rsplit("=", 1)[1]))
print(f"{len(users)} user deck tables")

sample = [users[i] for i in range(0, len(users), max(1, len(users) // 300))][:300]

tot_rows = tot_nonzero = tot_resolved = tot_selfparent = 0
root_vals = Counter()
depths = Counter()
cycles = 0
older_parent = younger_parent = 0
preset_same = preset_diff = 0
per_user_maxdepth = []
ms_like_parent = 0

for d in sample:
    df = pd.read_parquet(os.path.join(d, "data.parquet"))
    ids = set(df["deck_id"].tolist())
    par = dict(zip(df["deck_id"], df["parent_id"]))
    pres = dict(zip(df["deck_id"], df["preset_id"]))
    tot_rows += len(df)
    maxd = 0
    for did, pid in par.items():
        if pid == 0 or pd.isna(pid):
            root_vals[int(pid) if not pd.isna(pid) else -1] += 1
            continue
        tot_nonzero += 1
        if pid == did:
            tot_selfparent += 1
        if pid in ids:
            tot_resolved += 1
            if pid < did:
                older_parent += 1
            else:
                younger_parent += 1
            if pres.get(pid) == pres.get(did):
                preset_same += 1
            else:
                preset_diff += 1
        if 1_000_000_000_000 < pid < 2_000_000_000_000:  # plausible epoch-ms 2001..2033
            ms_like_parent += 1
        # walk to root
        seen, cur, depth = set(), did, 0
        while True:
            nxt = par.get(cur, 0)
            if nxt == 0 or nxt not in par:
                break
            if nxt in seen:
                cycles += 1
                break
            seen.add(nxt)
            cur = nxt
            depth += 1
            if depth > 50:
                cycles += 1
                break
        depths[depth] += 1
        maxd = max(maxd, depth)
    per_user_maxdepth.append(maxd)

print(f"sampled {len(sample)} users, {tot_rows} deck rows")
print(f"parent_id == 0 (root convention): {sum(root_vals.values())} rows; "
      f"other root-ish values: {[v for v in root_vals if v != 0][:5]}")
print(f"parent_id != 0: {tot_nonzero}  ({100*tot_nonzero/max(tot_rows,1):.1f}% of decks)")
print(f"  resolves to a deck_id in the SAME user's table: {tot_resolved} "
      f"({100*tot_resolved/max(tot_nonzero,1):.1f}%)")
print(f"  self-parent rows: {tot_selfparent}")
print(f"  parent id looks epoch-ms: {ms_like_parent}/{tot_nonzero}")
print(f"  parent OLDER than child (pid < did): {older_parent}; younger: {younger_parent}")
print(f"  child shares parent's preset: {preset_same}; differs: {preset_diff}")
print(f"cycles/over-deep walks: {cycles}")
print(f"depth histogram (levels above the deck): {dict(sorted(depths.items()))}")
print(f"max depth per user: mean {sum(per_user_maxdepth)/len(per_user_maxdepth):.2f}, "
      f"max {max(per_user_maxdepth)}")

# concrete example
ex = pd.read_parquet(os.path.join(sample[min(7, len(sample) - 1)], "data.parquet"))
print("\nexample user table (first 12 rows):")
print(ex.head(12).to_string(index=False))
