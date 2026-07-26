"""Do the FSRS-Anki-20k users overlap with anki-revlogs-10k?

Fingerprint = raw Anki card_id (epoch-ms card creation time). The -id build of the
10k set keeps them raw, and the 20k revlogs carry them raw too, so a shared card_id
means the same physical collection.

For each 20k user we have the FIRST entry's cid (scan20k.py). We ask whether that
card belongs to any 10k user.
"""
import os, csv, numpy as np, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
TEN = r"C:\Users\Andrew\anki-revlogs-10k-id\cards"

print("loading 10k card ids ...", flush=True)
ids, owners = [], []
dirs = sorted(os.listdir(TEN), key=lambda s: int(s.split("=")[1]))
for n, d in enumerate(dirs):
    uid = int(d.split("=")[1])
    t = pq.read_table(os.path.join(TEN, d, "data.parquet"), columns=["card_id"])
    a = t.column("card_id").to_numpy().astype(np.int64)
    ids.append(a)
    owners.append(np.full(a.shape, uid, dtype=np.int32))
    if (n + 1) % 2000 == 0:
        print(f"  {n+1}/{len(dirs)}", flush=True)

ids = np.concatenate(ids)
owners = np.concatenate(owners)
order = np.argsort(ids, kind="stable")
ids, owners = ids[order], owners[order]
print(f"10k: {len(ids):,} cards over {len(dirs):,} users", flush=True)

def _i64(s):
    # protobuf encodes negative int64 as a 10-byte unsigned varint (two's complement)
    v = int(s)
    return v - (1 << 64) if v >= (1 << 63) else v


rows = list(csv.DictReader(open(os.path.join(HERE, "first_entries_20k.csv"))))
cid = np.array([_i64(r["first_cid"]) for r in rows], dtype=np.int64)
print(f"20k: {len(cid):,} users fingerprinted", flush=True)

pos = np.searchsorted(ids, cid)
pos = np.clip(pos, 0, len(ids) - 1)
hit = ids[pos] == cid
print()
print(f"20k users whose first card is a known 10k card: {hit.sum():,} / {len(cid):,}"
      f"  ({100.0*hit.mean():.1f}%)")

if hit.any():
    matched10k = np.unique(owners[pos[hit]])
    print(f"distinct 10k users matched: {len(matched10k):,}")
    ev = matched10k[(matched10k >= 5001) & (matched10k <= 10000)]
    va = matched10k[(matched10k >= 5001) & (matched10k <= 7500)]
    print(f"  of which EVAL half   (5001-10000): {len(ev):,}")
    print(f"  of which VAL subset  (5001- 7500): {len(va):,}")
    print(f"  of which TRAIN       (   1- 5000): {len(matched10k[matched10k <= 5000]):,}")
    print(f"sample matches (20k shard/user -> 10k user): "
          f"{[(rows[i]['shard'], rows[i]['user'], int(owners[pos[i]])) for i in np.flatnonzero(hit)[:8]]}")
