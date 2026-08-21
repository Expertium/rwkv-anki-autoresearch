"""How predictable is the SIBLING GAP from the existing inputs -- per user, not pooled?

WHY THIS EXISTS. Two pooled runs disagreed badly on the same quantity: R2 of scaled_sibling_gap
from the 23 original per-row features came out 0.103 on one 4-user sample and 0.370 on a 12-user
sample. A 3.6x spread means the pooled number is not a property of the feature, it is a property of
whichever users landed in the sample.

AND POOLING IS THE WRONG QUESTION ANYWAY. A regression over concatenated users can exploit
BETWEEN-user structure -- if heavy users have both shorter sibling gaps and different elapsed-time
profiles, the fit picks that up. But the model does not predict across users; it carries a
per-user state. What matters is whether the gap is derivable WITHIN a user's own stream, which is
what a per-user regression measures.

Reports the per-user R2 distribution, plus the pooled value for contrast, plus the shuffled floor.

⚠ TRAIN-HALF USERS ONLY.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/sibling_gap_peruser.py [n_users]
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["RWKV_ID_FEATURES"] = "1"
sys.path.insert(0, os.getcwd())

from rwkv import id_features as idf  # noqa: E402
from rwkv.data_processing import CARD_FEATURE_COLUMNS, get_rwkv_data  # noqa: E402

IDD = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
UIDS = list(range(1, 5001, max(1, 5000 // N)))[:N]
NEW = list(idf.NEW_COLUMNS)
OLD = [c for c in CARD_FEATURE_COLUMNS if c not in NEW]
TGT = "scaled_sibling_gap"


def r2(X, y, seed=0, lam=10.0):
    if len(y) < 200:
        return None
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(y))
    X, y = X[p], y[p]
    k = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:k], X[k:], y[:k], y[k:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ym))
    pred = Xte @ w + ym
    sst = float(((yte - yte.mean()) ** 2).sum())
    if sst / max(len(yte), 1) < 1e-8:
        return None
    return 1.0 - float(((yte - pred) ** 2).sum()) / sst


import pandas as pd  # noqa: E402

per_user, cover, pooled_parts = [], [], []
for uid in UIDS:
    d = IDD / "revlogs" / ("user_id=%d" % uid)
    if not d.exists():
        continue
    df = get_rwkv_data(IDD, uid)
    if TGT not in df.columns:
        continue
    # get_rwkv_data DROPS review_time (it is a reject column), so the raw gap cannot be recomputed
    # from its output. The sentinel is recoverable instead: id_features writes exactly 0.0 for "no
    # preceding sibling", while a GENUINE zero-second gap standardizes to
    # (log(1+1e-5) - 9.3354) / 4.2198 = -2.212. So exact 0.0 identifies the sentinel, and a real
    # gap landing on exactly 0.0 would require _log_t(g) to equal the mean bit-for-bit.
    defined = df[TGT].to_numpy(dtype=np.float64) != 0.0
    cover.append(float(defined.mean()))
    # regress ONLY on rows where the gap is actually defined -- elsewhere it is the sentinel 0.0
    # and "predicting" a constant is not the question.
    sub = df.loc[defined, OLD + [TGT]].astype(np.float64)
    if len(sub) < 200:
        per_user.append((uid, None, int(defined.sum())))
        continue
    v = r2(sub[OLD].to_numpy(), sub[TGT].to_numpy())
    per_user.append((uid, v, int(defined.sum())))
    pooled_parts.append(sub)
    print("  user %-5d defined %6d (%.1f%%)  R2 %s"
          % (uid, int(defined.sum()), 100 * defined.mean(),
             "n/a" if v is None else "%+.4f" % v), flush=True)

vals = [v for _, v, _ in per_user if v is not None]
print("")
print("--- PER-USER R2 of %s from the %d original features, on DEFINED rows only" % (TGT, len(OLD)))
print("  users with enough defined rows : %d of %d" % (len(vals), len(per_user)))
if vals:
    a = np.array(vals)
    print("  median %+.4f   mean %+.4f   p10 %+.4f   p90 %+.4f   min %+.4f   max %+.4f"
          % (np.median(a), a.mean(), np.percentile(a, 10), np.percentile(a, 90), a.min(), a.max()))
print("  gap coverage per user: median %.3f  p10 %.3f  p90 %.3f"
      % (np.median(cover), np.percentile(cover, 10), np.percentile(cover, 90)))

if pooled_parts:
    pooled = pd.concat(pooled_parts, ignore_index=True)
    if len(pooled) > 120000:
        pooled = pooled.sample(n=120000, random_state=0).reset_index(drop=True)
    pv = r2(pooled[OLD].to_numpy(), pooled[TGT].to_numpy())
    sh = r2(pooled[OLD].to_numpy(), np.random.default_rng(1).permutation(pooled[TGT].to_numpy()))
    print("")
    print("  POOLED across users (%d rows): %+.4f    shuffled floor %+.4f" % (len(pooled), pv, sh))
    print("  If pooled >> the per-user median, the pooled fit is exploiting BETWEEN-user")
    print("  differences, which the model cannot use -- it carries a per-user state.")
