"""Overlap between FSRS-Anki-20k and anki-revlogs-10k, fingerprinted by REVIEW time.

card_id is useless as a fingerprint: shared decks carry their card ids to every
downloader, so one card id is the first card of 636 different 20k users.

A review's epoch-ms timestamp is local to the collection that performed it, so a
shared value means the same collection.  The 20k side gives the raw revlog `id`
(written on ANSWER); the -id build of the 10k set stores `review_time = id - duration`,
so the comparable value is `review_time + duration`.

We test membership of each 20k user's FIRST raw review in each 10k user's full
review set -- a LOWER bound on overlap, since that first entry may have been
dropped by the 10k build's filters (manual/filtered-deck rows, non-latest
learning sequences).
"""
import os, csv, numpy as np, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
TEN = r"C:\Users\Andrew\anki-revlogs-10k-id\revlogs"


def _i64(s):
    v = int(s)
    return v - (1 << 64) if v >= (1 << 63) else v


rows = list(csv.DictReader(open(os.path.join(HERE, "first_entries_20k.csv"))))
fid = np.array([_i64(r["first_id"]) for r in rows], dtype=np.int64)
order = np.argsort(fid, kind="stable")
fid_s = fid[order]
print(f"20k: {len(fid):,} users, {len(np.unique(fid)):,} distinct first review times", flush=True)

dirs = sorted(os.listdir(TEN), key=lambda s: int(s.split("=")[1]))
hits = {}          # 20k row index -> 10k user id
n_rev = 0
for n, d in enumerate(dirs):
    uid = int(d.split("=")[1])
    t = pq.read_table(os.path.join(TEN, d, "data.parquet"),
                      columns=["review_time", "duration"])
    raw = (t.column("review_time").to_numpy().astype(np.int64)
           + t.column("duration").to_numpy().astype(np.int64))
    n_rev += len(raw)
    pos = np.clip(np.searchsorted(fid_s, raw), 0, len(fid_s) - 1)
    m = fid_s[pos] == raw
    if m.any():
        for j in np.unique(pos[m]):
            hits.setdefault(int(order[j]), uid)
    if (n + 1) % 2000 == 0:
        print(f"  {n+1}/{len(dirs)}  ({n_rev:,} reviews, {len(hits)} matches)", flush=True)

print(f"\nscanned {n_rev:,} reviews over {len(dirs):,} 10k users")
print(f"20k users found inside the 10k set: {len(hits):,} / {len(fid):,} "
      f"({100.0*len(hits)/len(fid):.1f}%)   [lower bound]")

if hits:
    matched = np.array(sorted(set(hits.values())))
    print(f"distinct 10k users matched: {len(matched):,}")
    print(f"  TRAIN      (   1- 5000): {int(((matched >= 1) & (matched <= 5000)).sum()):,}")
    print(f"  EVAL half  (5001-10000): {int((matched >= 5001).sum()):,}")
    print(f"  VAL subset (5001- 7500): {int(((matched >= 5001) & (matched <= 7500)).sum()):,}")
    ex = [(rows[i]['shard'] + '/' + rows[i]['user'], u) for i, u in list(hits.items())[:8]]
    print(f"sample (20k shard/user -> 10k user): {ex}")
