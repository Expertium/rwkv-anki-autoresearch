import json
import lmdb

env = lmdb.open("F:/rwkv_lmdb/test_db_5k", map_size=250_000_000_000, readonly=True,
                lock=False, readahead=False)
sizes = {}
with env.begin() as txn:
    for uid in range(5001, 5061):
        raw = txn.get(f"{uid}_batches".encode())
        if raw is None:
            continue
        b = json.loads(raw)
        sizes[uid] = (max(x[2] for x in b), len(b))
env.close()
for uid, (sz, nb) in sorted(sizes.items(), key=lambda kv: kv[1][0])[:8]:
    print(f"user {uid}: work {sz:,} batches {nb}")
print("...")
for uid, (sz, nb) in sorted(sizes.items(), key=lambda kv: kv[1][0])[25:29]:
    print(f"user {uid}: work {sz:,} batches {nb}")
