import os, sys, json
os.environ.setdefault("RWKV_ID_FEATURES", "1"); os.environ.setdefault("RWKV_REAL_CYCLES", "1")
sys.path.insert(0, os.path.abspath("."))
import numpy as np, lmdb
from rwkv.data_processing import CARD_FEATURE_COLUMNS
from rwkv.utils import load_tensor
COLS = list(CARD_FEATURE_COLUMNS)
DB = "F:/rwkv_lmdb/train_db_5k_h1_id5"
USERS = [1, 17, 101, 333, 477, 1503, 2501, 3777]
env = lmdb.open(DB, readonly=True, lock=False, max_readers=4)
per = {}
Xs, Ss = [], []
with env.begin() as txn:
    for u in USERS:
        raw = txn.get(f"{u}_batches".encode())
        if raw is None: continue
        for (s, e, L) in sorted(json.loads(raw))[:2]:
            p = f"{u}_{s}-{e}_{L}_"
            cf = load_tensor(txn, p + "card_features", "cpu").float().numpy()
            sk = load_tensor(txn, p + "skips", "cpu").numpy().astype(bool)
            Xs.append(cf); Ss.append(sk)
            per.setdefault(u, []).append(cf[~sk])
env.close()
X = np.concatenate(Xs, 0); S = np.concatenate(Ss, 0); R = X[~S]
ix = {c: i for i, c in enumerate(COLS)}

print("== the three *_is_nan columns: are they the same column? ==")
a, b, c = R[:, ix["note_id_is_nan"]], R[:, ix["deck_id_is_nan"]], R[:, ix["preset_id_is_nan"]]
print(f"  note==deck {np.mean(a==b)*100:.3f}%   note==preset {np.mean(a==c)*100:.3f}%   "
      f"deck==preset {np.mean(b==c)*100:.3f}%")
print("  per-user NaN-metadata rate (note_id_is_nan mean):")
for u, arrs in per.items():
    r = np.concatenate(arrs, 0)
    print(f"    user {u:>5}: note {r[:, ix['note_id_is_nan']].mean()*100:6.2f}%  "
          f"deck {r[:, ix['deck_id_is_nan']].mean()*100:6.2f}%  "
          f"sib-sentinel {np.mean(r[:, ix['scaled_sibling_gap']] == 0.0)*100:6.2f}%  "
          f"tsince-clip {np.mean(r[:, ix['scaled_t_since_any_review']] <= -1.66)*100:6.2f}%")

for name in ("scaled_t_since_any_review", "scaled_sibling_gap", "scaled_deck_age_at_review",
             "scaled_creation_to_first_review"):
    v = R[:, ix[name]]
    vals, cnt = np.unique(v, return_counts=True)
    o = np.argsort(-cnt)[:4]
    print(f"\n== {name}: min {v.min():.4f}  top values ==")
    for k in o:
        print(f"    {vals[k]:+.5f}  {100*cnt[k]/v.size:6.2f}%"
              + ("   <-- equals min (clip/sentinel)" if vals[k] == v.min() else ""))
