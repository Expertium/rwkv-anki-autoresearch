"""How degenerate is the PRESET pooling level vs a would-be parent/root-DECK level?
Cheap version: decks tables only (all decks a user owns), 800 users."""
import glob
import os

import pandas as pd

PUB = r"C:\Users\Andrew\anki-revlogs-10k"
users = sorted(glob.glob(os.path.join(PUB, "decks", "user_id=*")),
               key=lambda p: int(p.rsplit("=", 1)[1]))
sample = [users[i] for i in range(0, len(users), max(1, len(users) // 800))][:800]

rows = []
for u in sample:
    df = pd.read_parquet(os.path.join(u, "data.parquet"))
    ids = set(df["deck_id"])
    par = dict(zip(df["deck_id"], df["parent_id"]))

    def root_of(did):
        cur, seen = did, set()
        while True:
            p = par.get(cur)
            if p is None or p not in par or p in seen:
                return cur
            seen.add(p)
            cur = p

    roots = {root_of(d) for d in ids}
    rows.append({"decks": len(ids), "roots": len(roots),
                 "presets": df["preset_id"].nunique()})

df = pd.DataFrame(rows)
print(f"{len(df)} users (all owned decks)")
print(df.describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
for col in ("presets", "roots", "decks"):
    print(f"share of users with exactly 1 distinct {col}: {(df[col] == 1).mean():.1%}")
print(f"share where roots > presets (tree is a FINER pooling level): "
      f"{(df.roots > df.presets).mean():.1%}")
print(f"share where decks > roots > 1 (tree adds a real middle level): "
      f"{((df.decks > df.roots) & (df.roots > 1)).mean():.1%}")
