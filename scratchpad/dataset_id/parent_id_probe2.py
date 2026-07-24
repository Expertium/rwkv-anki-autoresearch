"""Follow-up: (a) does the PUBLISHED anonymized anki-revlogs-10k also carry parent_id?
(b) how useful is the deck tree as a POOLING LEVEL for the 5-stream hierarchy --
i.e. per user, how many distinct decks / root-decks / presets are actually USED by
reviewed cards, and does root-deck grouping differ from preset grouping?"""
import glob
import os

import pandas as pd

PUB = r"C:\Users\Andrew\anki-revlogs-10k"
IDD = r"C:\Users\Andrew\anki-revlogs-10k-id"

# (a) published dataset schema
pub_users = sorted(glob.glob(os.path.join(PUB, "decks", "user_id=*")),
                   key=lambda p: int(p.rsplit("=", 1)[1]))
print(f"published decks tables: {len(pub_users)}")
if pub_users:
    d = pd.read_parquet(os.path.join(pub_users[0], "data.parquet"))
    print(f"published columns: {list(d.columns)}")
    print(d.head(8).to_string(index=False))
    if "parent_id" in d.columns:
        nz = (d["parent_id"] != 0).mean()
        print(f"published parent_id non-zero share (user {pub_users[0][-4:]}): {nz:.2f}")

# (b) pooling-level statistics on the id dataset (needs cards+decks+revlogs)
id_users = sorted(glob.glob(os.path.join(IDD, "decks", "user_id=*")),
                  key=lambda p: int(p.rsplit("=", 1)[1]))
sample = [id_users[i] for i in range(0, len(id_users), max(1, len(id_users) // 120))][:120]
rows = []
for du in sample:
    uid = du.rsplit("=", 1)[1]
    cpath = os.path.join(IDD, "cards", f"user_id={uid}", "data.parquet")
    rpath = os.path.join(IDD, "revlogs", f"user_id={uid}", "data.parquet")
    if not (os.path.exists(cpath) and os.path.exists(rpath)):
        continue
    decks = pd.read_parquet(os.path.join(du, "data.parquet"))
    cards = pd.read_parquet(cpath)
    rev = pd.read_parquet(rpath, columns=["card_id"])
    par = dict(zip(decks["deck_id"], decks["parent_id"]))
    pres = dict(zip(decks["deck_id"], decks["preset_id"]))

    def root_of(did, _cache={}):
        cur, seen = did, set()
        while True:
            p = par.get(cur, 0)
            if p == 0 or p not in par or p in seen:
                return cur
            seen.add(p)
            cur = p

    def depth_of(did):
        cur, dep, seen = did, 0, set()
        while True:
            p = par.get(cur, 0)
            if p == 0 or p not in par or p in seen:
                return dep
            seen.add(p)
            cur = p
            dep += 1

    reviewed_cards = set(rev["card_id"].unique().tolist())
    cc = cards[cards["card_id"].isin(reviewed_cards)] if "card_id" in cards.columns else cards
    used_decks = cc["deck_id"].unique().tolist()
    if not len(used_decks):
        continue
    roots = {root_of(d) for d in used_decks}
    presets = {pres.get(d, 0) for d in used_decks}
    deps = [depth_of(d) for d in used_decks]
    rows.append({
        "user": int(uid),
        "cards_reviewed": len(cc),
        "decks_used": len(used_decks),
        "root_decks": len(roots),
        "presets": len(presets),
        "mean_depth": sum(deps) / len(deps),
        "max_depth": max(deps),
        "nested_share": sum(1 for x in deps if x > 0) / len(deps),
    })

df = pd.DataFrame(rows)
print(f"\n(b) pooling levels over {len(df)} sampled users (decks actually reviewed):")
print(df[["cards_reviewed", "decks_used", "root_decks", "presets",
          "mean_depth", "max_depth", "nested_share"]].describe().round(2).to_string())
print(f"\nusers where root-deck grouping is COARSER than deck (root < decks): "
      f"{(df.root_decks < df.decks_used).mean():.0%}")
print(f"users where root-deck count differs from preset count: "
      f"{(df.root_decks != df.presets).mean():.0%}")
print(f"users with any nested deck among reviewed decks: {(df.max_depth > 0).mean():.0%}")
