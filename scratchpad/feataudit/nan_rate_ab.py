"""Does the -id build LOSE card metadata the published build has?

note_id_is_nan is column 13 in both layouts (it precedes every -id change). Compare the same
users' rate in gen-5 (-id) against the published e2s db, on the FIRST chunk of each user so the
row sets are comparable in size.
"""
import os, sys, json
sys.path.insert(0, os.path.abspath("."))
import numpy as np, lmdb
from rwkv.utils import load_tensor

IDX = 13
DBS = {"published_e2s": "F:/rwkv_lmdb/train_db_5k_h1_e2s",
       "gen5_id5": "F:/rwkv_lmdb/train_db_5k_h1_id5"}
USERS = list(range(1, 5000, 157))
out = {}
for tag, path in DBS.items():
    env = lmdb.open(path, readonly=True, lock=False, max_readers=4)
    tot = nan = 0
    per = []
    with env.begin() as txn:
        for u in USERS:
            raw = txn.get(f"{u}_batches".encode())
            if raw is None:
                continue
            s, e, L = sorted(json.loads(raw))[0]
            p = f"{u}_{s}-{e}_{L}_"
            cf = load_tensor(txn, p + "card_features", "cpu").float().numpy()
            sk = load_tensor(txn, p + "skips", "cpu").numpy().astype(bool)
            v = cf[~sk][:, IDX]
            tot += v.size; nan += int(v.sum())
            per.append((u, v.mean()))
    env.close()
    out[tag] = (tot, nan, dict(per))
    print(f"{tag:<16} users {len(per):>3}  rows {tot:>9}  note_id_is_nan {100*nan/tot:6.2f}%")

a, b = out["published_e2s"][2], out["gen5_id5"][2]
common = sorted(set(a) & set(b))
d = np.array([b[u] - a[u] for u in common])
print(f"\npaired over {len(common)} users: mean diff {100*d.mean():+.3f} pp, "
      f"max {100*d.max():+.2f}, min {100*d.min():+.2f}")
worst = sorted(common, key=lambda u: -(b[u] - a[u]))[:6]
for u in worst:
    print(f"   user {u:>5}: published {100*a[u]:6.2f}%  gen5 {100*b[u]:6.2f}%")
