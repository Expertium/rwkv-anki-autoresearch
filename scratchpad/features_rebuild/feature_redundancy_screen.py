"""Which of the 23 new columns carry information the model did not already have?

WHY, IN GPU TERMS. If the bundle wins, family-level ablation is 7 arms at ~7.75 h = ~54 h. This
screen costs minutes of CPU and says which families could possibly matter, so the arms can be spent
on those instead of swept blindly. Same discipline as the sibling redundancy screen, which produced
a real finding (the note stream already reconstructs ~31% of the sibling gap's variance).

WHAT IT MEASURES, per new column, held out 70/30:
  R2_old  -- predictability from the 23 ORIGINAL per-row features alone. High means the column is a
             function of what the model already sees on the same row, so it adds nothing new.
  R2_all  -- predictability from the originals PLUS the other 22 new columns. The GAP between this
             and R2_old is intra-family redundancy: how much a column is explained by its siblings
             in the bundle, which is exactly what a FAMILY-level ablation would remove together.
  R2_shuf -- the same regression against a shuffled target. The overfitting floor at this
             dimensionality; any R2 not clearly above it means nothing.

WHAT IT CANNOT SEE, stated so the result is not over-read. This is a PER-ROW test. It cannot tell
whether the RECURRENCE could derive a column over time from its state -- that needs a state dump,
which is what sibling_redundancy_screen.py does for one column. So a LOW R2 here means "not
trivially redundant with the current row"; it does NOT prove the information is unavailable to the
model. A HIGH R2 is the decisive direction: it means the column is already there.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/feature_redundancy_screen.py [n_users]
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
N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
UIDS = [1, 101, 209, 417, 625, 833][:N_USERS]

NEW = list(idf.NEW_COLUMNS)
OLD = [c for c in CARD_FEATURE_COLUMNS if c not in NEW]

FAMILY = {
    "time-of-day": ["tod_sin", "tod_cos", "tod_dev_sin", "tod_dev_cos"],
    "calendar": ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"],
    "recency+ages": ["scaled_t_since_any_review", "scaled_user_tenure",
                     "scaled_creation_to_first_review"],
    "deck": ["scaled_deck_age_at_review", "card_predates_deck", "is_default_deck",
             "scaled_deck_depth"],
    "creation-batch": ["scaled_creation_batch_1min", "scaled_creation_batch_1h",
                       "scaled_creation_batch_1d", "scaled_creation_batch_pos_1h"],
    "preset": ["is_default_preset"],
    "the two omissions": ["scaled_sibling_gap", "card_predates_first_review"],
}


def ridge_r2(X, y, seed=0, lam=10.0):
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(y))
    X, y = X[p], y[p]
    k = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:k], X[k:], y[:k], y[k:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ym = ytr.mean()
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (ytr - ym))
    pred = Xte @ w + ym
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    # ⚠ R2 is UNDEFINED for a near-constant target: ss_tot -> 0 makes the ratio explode. The first
    # run of this screen printed R2 = -2.0e9 for is_default_deck (std 0.006) and that number is
    # meaningless, not "very bad". Refuse rather than clamp -- a clamped -1e9 would still be read
    # as a value. The near-constancy is itself the finding for such a column.
    if ss_tot / max(len(yte), 1) < 1e-8:
        return float("nan")
    return 1.0 - ss_res / ss_tot


frames = []
for uid in UIDS:
    df = get_rwkv_data(IDD, uid)
    frames.append(df[[c for c in CARD_FEATURE_COLUMNS if c in df.columns]].astype(np.float64))
    print("  loaded user %d (%d rows)" % (uid, len(df)), flush=True)

import pandas as pd  # noqa: E402
data = pd.concat(frames, ignore_index=True)
# subsample: 23 ridge solves on hundreds of thousands of rows is pointless precision
if len(data) > 60000:
    data = data.sample(n=60000, random_state=0).reset_index(drop=True)
print("\nregression matrix: %d rows, %d old cols, %d new cols\n" % (len(data), len(OLD), len(NEW)))

Xold = data[OLD].to_numpy()
results = {}
for c in NEW:
    y = data[c].to_numpy()
    if float(y.std()) < 1e-9:
        results[c] = (float("nan"), float("nan"), float("nan"), 0.0)
        continue
    others = [k for k in NEW if k != c]
    Xall = np.concatenate([Xold, data[others].to_numpy()], axis=1)
    r_old = ridge_r2(Xold, y)
    r_all = ridge_r2(Xall, y)
    r_shuf = ridge_r2(Xall, np.random.default_rng(1).permutation(y))
    results[c] = (r_old, r_all, r_shuf, float(y.std()))

print("%-32s %8s %8s %8s %8s" % ("column", "R2_old", "R2_all", "R2_shuf", "std"))
print("-" * 70)
for fam, cols in FAMILY.items():
    print("[%s]" % fam)
    for c in cols:
        if c not in results:
            continue
        ro, ra, rs, sd = results[c]
        if np.isnan(ro) or np.isnan(ra):
            print("  %-30s %8s %8s %8s %8.3f   NEAR-CONSTANT on these users: R2 undefined,"
                  % (c, "n/a", "n/a", "n/a", sd))
            print("  %-30s %s   and a column with no variance carries nothing regardless"
                  % ("", " " * 26))
            continue
        print("  %-30s %+8.4f %+8.4f %+8.4f %8.3f" % (c, ro, ra, rs, sd))
    print("")

print("READ:")
print("  R2_old high  -> already a function of the CURRENT ROW's existing features; adds nothing.")
print("  R2_all high but R2_old low -> explained by its SIBLINGS in the bundle, i.e. the family")
print("     carries the information but the individual column is redundant WITHIN it. That is an")
print("     argument for family-level ablation over per-feature arms.")
print("  both low     -> genuinely new per-row information. NOT proof it helps, and NOT proof the")
print("     recurrence could not derive it over time -- only that the current row does not contain")
print("     it. Use sibling_redundancy_screen.py's state regression for the stronger claim.")
