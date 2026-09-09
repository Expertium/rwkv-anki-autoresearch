"""Do the gen-5 TRAIN and TEST databases agree on the input coding?

They are built by separate runs of data_processing, so a flag or constant that reached one and not
the other is a silent train/eval mismatch that no gate scores.
"""
import os, sys, json
os.environ.setdefault("RWKV_ID_FEATURES", "1"); os.environ.setdefault("RWKV_REAL_CYCLES", "1")
sys.path.insert(0, os.path.abspath("."))
import numpy as np, lmdb
from rwkv.data_processing import CARD_FEATURE_COLUMNS
from rwkv.utils import load_tensor
COLS = list(CARD_FEATURE_COLUMNS)

def grab(path, users):
    env = lmdb.open(path, readonly=True, lock=False, max_readers=4)
    Xs, Ss, n = [], [], 0
    with env.begin() as txn:
        for u in users:
            raw = txn.get(f"{u}_batches".encode())
            if raw is None: continue
            s, e, L = sorted(json.loads(raw))[0]
            p = f"{u}_{s}-{e}_{L}_"
            Xs.append(load_tensor(txn, p + "card_features", "cpu").float().numpy())
            Ss.append(load_tensor(txn, p + "skips", "cpu").numpy().astype(bool))
            n += 1
    env.close()
    X = np.concatenate(Xs, 0); S = np.concatenate(Ss, 0)
    return X[~S], n

TR, ntr = grab("F:/rwkv_lmdb/train_db_5k_h1_id5", range(1, 5000, 311))
TE, nte = grab("F:/rwkv_lmdb/test_db_5k_id5", range(5001, 7500, 156))
print(f"train {ntr} users {TR.shape}   test {nte} users {TE.shape}")
assert TR.shape[1] == TE.shape[1] == len(COLS), "WIDTH MISMATCH between train and test"
print("width identical and equal to CARD_FEATURE_COLUMNS: OK\n")

print(f"{'column':<30} {'train mean':>11} {'test mean':>11} {'tr std':>8} {'te std':>8} {'flag':>6}")
bad = 0
for j, c in enumerate(COLS):
    a, b = TR[:, j], TE[:, j]
    # a coding difference shows as a support difference, not a small mean shift between user halves
    supp = (a.min() > b.max()) or (b.min() > a.max())
    zero_a, zero_b = (a == 0).mean(), (b == 0).mean()
    flag = ""
    if supp: flag = "SUPPORT"
    elif abs(zero_a - zero_b) > 0.25: flag = "SENTINEL"
    elif a.std() > 0 and b.std() > 0 and (max(a.std(), b.std()) / min(a.std(), b.std())) > 3:
        flag = "SPREAD"
    if flag:
        bad += 1
        print(f"{c:<30} {a.mean():>11.4f} {b.mean():>11.4f} {a.std():>8.4f} {b.std():>8.4f} {flag:>8}")
print(f"\ncolumns flagged: {bad} of {len(COLS)}")
