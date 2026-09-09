"""Dead dims and duplicate dims in the gen-5 input vector, over ALL rows (real+query+probe)."""
import os, sys, json
os.environ.setdefault("RWKV_ID_FEATURES", "1"); os.environ.setdefault("RWKV_REAL_CYCLES", "1")
sys.path.insert(0, os.path.abspath("."))
import numpy as np, lmdb, torch
from rwkv.data_processing import CARD_FEATURE_COLUMNS
from rwkv.utils import load_tensor
COLS = list(CARD_FEATURE_COLUMNS)
env = lmdb.open("F:/rwkv_lmdb/train_db_5k_h1_id5", readonly=True, lock=False, max_readers=4)
Xs = []
with env.begin() as txn:
    for u in range(1, 5000, 311):
        raw = txn.get(f"{u}_batches".encode())
        if raw is None: continue
        s, e, L = sorted(json.loads(raw))[0]
        Xs.append(load_tensor(txn, f"{u}_{s}-{e}_{L}_card_features", "cpu").float().numpy())
env.close()
X = np.concatenate(Xs, 0)
print(f"rows {X.shape[0]}  dims {X.shape[1]}")
dead = [COLS[j] for j in range(X.shape[1]) if X[:, j].std() == 0]
print(f"\nCONSTANT over all rows: {dead if dead else 'none'}")

print("\nEXACTLY IDENTICAL column pairs:")
seen = {}
for j in range(X.shape[1]):
    key = X[:, j].tobytes()
    seen.setdefault(key, []).append(COLS[j])
for k, g in seen.items():
    if len(g) > 1:
        print("   " + " == ".join(g))

Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
C = np.corrcoef(Z.T)
np.fill_diagonal(C, 0)
print("\n|corr| > 0.95 pairs:")
ii, jj = np.where(np.triu(np.abs(C), 1) > 0.95)
for a, b in zip(ii, jj):
    print(f"   {COLS[a]:<30} {COLS[b]:<30} r={C[a,b]:+.4f}")

# what the model was actually built with
import glob
ck = sorted(glob.glob("scratchpad/ws10/w10_ws_109350.pth"))
if ck:
    sd = torch.load(ck[0], map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    for k in sd:
        if "features2card" in k and k.endswith("0.weight"):
            print(f"\ncheckpoint {ck[0]}: {k} shape {tuple(sd[k].shape)}")
            break
