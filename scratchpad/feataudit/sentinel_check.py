"""Can the model TELL that scaled_sibling_gap is undefined? And how stale are the
normalization constants after the 2026-08-19 end-to-start change?"""
import os, sys, json
os.environ.setdefault("RWKV_ID_FEATURES", "1"); os.environ.setdefault("RWKV_REAL_CYCLES", "1")
sys.path.insert(0, os.path.abspath("."))
import numpy as np, lmdb
from rwkv.data_processing import CARD_FEATURE_COLUMNS
from rwkv.id_features import STATISTICS_ID
from rwkv.utils import load_tensor
COLS = list(CARD_FEATURE_COLUMNS); ix = {c: i for i, c in enumerate(COLS)}
env = lmdb.open("F:/rwkv_lmdb/train_db_5k_h1_id5", readonly=True, lock=False, max_readers=4)
Xs, Ss = [], []
with env.begin() as txn:
    for u in range(1, 5000, 311):
        raw = txn.get(f"{u}_batches".encode())
        if raw is None: continue
        s, e, L = sorted(json.loads(raw))[0]
        p = f"{u}_{s}-{e}_{L}_"
        Xs.append(load_tensor(txn, p + "card_features", "cpu").float().numpy())
        Ss.append(load_tensor(txn, p + "skips", "cpu").numpy().astype(bool))
env.close()
R = np.concatenate(Xs, 0)[~np.concatenate(Ss, 0)]

und = (R[:, ix["scaled_sibling_gap"]] == 0.0).astype(float)
print(f"sibling gap UNDEFINED on {100*und.mean():.2f}% of real rows "
      f"(sentinel 0.0 sits inside the defined range "
      f"[{R[:, ix['scaled_sibling_gap']].min():.3f}, {R[:, ix['scaled_sibling_gap']].max():.3f}])")
best = []
for j, c in enumerate(COLS):
    v = R[:, j]
    if v.std() == 0 or c == "scaled_sibling_gap": continue
    best.append((abs(np.corrcoef(und, v)[0, 1]), c))
best.sort(reverse=True)
print("  best existing proxy for 'undefined':")
for r, c in best[:3]:
    print(f"    {c:<28} |r| = {r:.3f}")

print("\nnormalization constants vs what the gen-5 db actually contains "
      "(standardized column should be ~N(0,1)):")
for c in ("scaled_t_since_any_review", "scaled_user_tenure", "scaled_sibling_gap",
          "scaled_creation_to_first_review", "scaled_deck_age_at_review",
          "scaled_creation_batch_1h", "scaled_deck_depth"):
    v = R[:, ix[c]]
    print(f"  {c:<32} mean {v.mean():+7.3f}  std {v.std():6.3f}  min {v.min():+8.3f}")
