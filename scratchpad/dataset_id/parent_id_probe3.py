"""Does the PUBLISHED (anonymized/factorized) anki-revlogs-10k preserve a usable deck
tree? Factorization maps ids to small ints; if deck_id and parent_id were factorized
with the SAME codebook, parent_id still resolves to a row in the same table and the
hierarchy survives -- meaning deck-tree features need NO new dataset export."""
import glob
import os
from collections import Counter

import pandas as pd

PUB = r"C:\Users\Andrew\anki-revlogs-10k"
users = sorted(glob.glob(os.path.join(PUB, "decks", "user_id=*")),
               key=lambda p: int(p.rsplit("=", 1)[1]))
sample = [users[i] for i in range(0, len(users), max(1, len(users) // 200))][:200]

tot = res = 0
selfp = 0
root_code = Counter()
depths = Counter()
cycles = 0
for u in sample:
    df = pd.read_parquet(os.path.join(u, "data.parquet"))
    ids = set(df["deck_id"].tolist())
    par = dict(zip(df["deck_id"], df["parent_id"]))
    tot += len(df)
    res += sum(1 for p in df["parent_id"] if p in ids)
    selfp += sum(1 for d, p in par.items() if d == p)
    # the "root" sentinel = whatever code the unresolvable parents share
    for p in df["parent_id"]:
        if p not in ids:
            root_code[int(p)] += 1
    for did in par:
        cur, dep, seen = did, 0, set()
        while True:
            p = par.get(cur, None)
            if p is None or p not in par or p in seen or p == cur:
                break
            seen.add(p)
            cur = p
            dep += 1
            if dep > 60:
                cycles += 1
                break
        depths[dep] += 1

print(f"published: {len(sample)} users, {tot} deck rows")
print(f"parent_id resolves to a deck_id in the same table: {res} ({100*res/tot:.1f}%)")
print(f"self-parent rows (root sentinel encoded as self?): {selfp}")
print(f"unresolvable parent codes (root sentinel candidates): "
      f"{dict(list(root_code.most_common(5)))} total {sum(root_code.values())}")
print(f"depth histogram: {dict(sorted(depths.items())[:12])}")
print(f"cycles: {cycles}")

d0 = pd.read_parquet(os.path.join(sample[3], "data.parquet"))
print(f"\nexample published deck table (user {sample[3].rsplit('=',1)[1]}):")
print(d0.head(15).to_string(index=False))
print(f"deck_id range {d0.deck_id.min()}..{d0.deck_id.max()}, "
      f"parent_id range {d0.parent_id.min()}..{d0.parent_id.max()}")
