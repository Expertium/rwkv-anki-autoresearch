"""FAMILY-LEVEL predictability: how much of each new feature family is already implied by the old
features, and by the other new families?

Andrew 2026-08-21: "make a table that shows predictability between (families of) new features and
old features".

WHY FAMILIES AND NOT COLUMNS. The per-column screen showed R2 of 0.6-0.95 when a column is
predicted from its own siblings, so a single-column ablation would read null even for a column that
matters -- its information stays in the family. The family is therefore the unit that a GPU arm can
actually remove, and the unit this table should be about.

THE AGGREGATE. For a target family with k columns, each column is z-scored first (so a wide column
cannot dominate a narrow one), then

    R2_family = 1 - sum_k SS_res(k) / sum_k SS_tot(k)

on a held-out 30%. Equal weight per column, and it degrades to the ordinary R2 for k = 1.

READ THE MATRIX AS: "if I delete this family, how much of it could the model reconstruct from the
predictor set?" High = the family is redundant given that set. The OLD column is the one that says
whether a family was worth adding at all; the ALL-NEW column is what a family-level ablation is up
against.

⚠ PER-ROW ONLY. This cannot see what the RECURRENCE derives over time from its state -- see
sibling_redundancy_screen.py, which found the note stream already reconstructs ~31% of the sibling
gap. A LOW number here means "not trivially present in the current row", NOT "unavailable to the
model". A HIGH number is the decisive direction.

⚠ TRAIN-HALF USERS ONLY (1-5000). Descriptive though this is, deriving anything from 5001-10000
would put eval-set structure into a decision.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/family_predictability.py [n_users]
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
N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
TRAIN_LO, TRAIN_HI = 1, 5000
stride = max(1, (TRAIN_HI - TRAIN_LO + 1) // N_USERS)
UIDS = list(range(TRAIN_LO, TRAIN_HI + 1, stride))[:N_USERS]

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
    # ⚠ NOT an "omissions" family. That label named MY process (features I forgot to implement),
    # not the features' meaning, and it hid Andrew's sibling gap inside a bookkeeping bucket. The
    # two are also semantically unrelated -- note-level interference vs collection provenance --
    # so grouping them measured nothing. Each stands alone.
    "sibling-gap": ["scaled_sibling_gap"],
    "card-predates-1st": ["card_predates_first_review"],
}
SHORT = {"time-of-day": "tod", "calendar": "cal", "recency+ages": "recy", "deck": "deck",
         "creation-batch": "batch", "preset": "prst", "sibling-gap": "sib",
         "card-predates-1st": "cp1st"}


def family_r2(X, Y, seed=0, lam=10.0):
    """Held-out aggregate R2 for a MULTI-column target Y. Columns are z-scored so a wide column
    cannot dominate. Returns nan if the target has no usable variance."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(X.shape[0])
    X, Y = X[p], Y[p]
    k = int(0.7 * X.shape[0])
    Xtr, Xte, Ytr, Yte = X[:k], X[k:], Y[:k], Y[k:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ymu, ysd = Ytr.mean(0), Ytr.std(0)
    keep = ysd > 1e-6                      # drop dead columns from the aggregate, and say so
    if not keep.any():
        return float("nan"), 0
    Ytr = (Ytr[:, keep] - ymu[keep]) / ysd[keep]
    Yte = (Yte[:, keep] - ymu[keep]) / ysd[keep]
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ W
    ss_res = float(((Yte - pred) ** 2).sum())
    ss_tot = float(((Yte - Yte.mean(0)) ** 2).sum())
    if ss_tot <= 0:
        return float("nan"), int(keep.sum())
    return 1.0 - ss_res / ss_tot, int(keep.sum())


frames = []
for uid in UIDS:
    d = IDD / "revlogs" / ("user_id=%d" % uid)
    if not d.exists():
        continue
    df = get_rwkv_data(IDD, uid)
    frames.append(df[[c for c in CARD_FEATURE_COLUMNS if c in df.columns]].astype(np.float32))
    print("  loaded user %d (%d rows)" % (uid, len(df)), flush=True)

import pandas as pd  # noqa: E402
data = pd.concat(frames, ignore_index=True)
del frames
if len(data) > 120000:
    data = data.sample(n=120000, random_state=0).reset_index(drop=True)
print("\nmatrix: %d rows from %d users, %d old cols, %d new cols\n"
      % (len(data), len(UIDS), len(OLD), len(NEW)))

# ---- variance audit: a dead column cannot carry anything, whatever its redundancy ----
print("--- VARIANCE AUDIT (a near-constant column is dead regardless of redundancy)")
dead = []
for c in NEW:
    sd = float(data[c].std())
    if sd < 0.01:
        dead.append((c, sd))
for c, sd in sorted(dead, key=lambda x: x[1]):
    print("  %-32s std %.5f   NEAR-CONSTANT across %d users" % (c, sd, len(UIDS)))
if not dead:
    print("  none: every new column has usable variance across these users")
print("")

Xold = data[OLD].to_numpy(dtype=np.float64)
fam_names = list(FAMILY)
cols_hdr = ["OLD"] + [SHORT[f] for f in fam_names] + ["ALLNEW", "OLD+NEW"]

print("--- FAMILY PREDICTABILITY (held-out aggregate R2; rows = family being predicted)")
print("%-16s %6s | %s | %7s %8s" % ("target family", "OLD",
                                    " ".join("%6s" % SHORT[f] for f in fam_names),
                                    "ALLNEW", "OLD+NEW"))
print("-" * 96)
rows = {}
for tgt in fam_names:
    tcols = [c for c in FAMILY[tgt] if c in data.columns]
    Y = data[tcols].to_numpy(dtype=np.float64)
    cells = []
    r_old, nk = family_r2(Xold, Y)
    cells.append(r_old)
    for src in fam_names:
        if src == tgt:
            cells.append(None)
            continue
        Xs = data[[c for c in FAMILY[src] if c in data.columns]].to_numpy(dtype=np.float64)
        r, _ = family_r2(Xs, Y)
        cells.append(r)
    other = [c for c in NEW if c not in tcols]
    r_all, _ = family_r2(data[other].to_numpy(dtype=np.float64), Y)
    r_both, _ = family_r2(np.concatenate([Xold, data[other].to_numpy(dtype=np.float64)], axis=1), Y)
    cells += [r_all, r_both]
    rows[tgt] = cells

    def fmt(v):
        if v is None:
            return "     -"
        return "   nan" if np.isnan(v) else "%+6.3f" % v
    print("%-16s %s | %s | %s %s"
          % (tgt, fmt(cells[0]), " ".join(fmt(v) for v in cells[1:1 + len(fam_names)]),
             fmt(cells[-2]), fmt(cells[-1])))
    if nk < len(tcols):
        print("%-16s   (%d of %d columns had usable variance; the rest were dropped)"
              % ("", nk, len(tcols)))

print("")
print("READ: 'if this family were deleted, how much could be reconstructed from that predictor set?'")
print("  OLD high     -> the family was already implied by the ORIGINAL inputs; adding it was")
print("                  unlikely to buy anything.")
print("  ALLNEW high  -> a family-level ablation would still leave the information in the bundle,")
print("                  so even a FAMILY arm may read null. Those families are entangled and")
print("                  should be ablated together or not at all.")
print("  both low     -> the family is the only carrier of its information. These are the arms")
print("                  worth GPU time.")
print("PER-ROW ONLY -- this cannot see what the recurrence derives over time from its state.")
