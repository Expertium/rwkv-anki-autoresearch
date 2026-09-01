"""Is the fixc-vs-iter53 regression caused by the note-placeholder change?

THE PUZZLE. fixc carries the Bug A and Bug C id fixes and lands +0.000141 ahead / +0.000084 imm
WORSE than iter 53, while the same class of fix measured +0.000148 / +0.000169 BETTER on the
KD-off featA/featA2 pair. Two explanations were on the table and the arms differ in two ways, so
neither is attributable from the means alone:

  H1  THE NOTE-POOLING DOSE. The three databases pool NaN-note cards differently:
        iter 53 (train_db_5k_h1)      Bug A + Bug C  -> ALL such cards share ONE note
        featA2  (train_db_5k_h1_fix)  Bug A fixed    -> ~812 notes over 49,186 cards (~60 each)
        fixc    (train_db_5k_h1_fixc) both fixed     -> ONE NOTE PER CARD (verified ratio 1.0000)
      So featA->featA2 is partial de-pooling and iter53->fixc is total de-pooling. If the note
      stream needs SOME cross-card evidence, partial helps and total hurts -- a U-shape.
  H2  THE KD REGIME. featA/featA2 are KD-off, iter53/fixc are KD-on, and the fixes simply do not
      pay when a teacher is supplying most of the target.

THE DISCRIMINATOR. H1 is a claim about WHICH REVIEWS are affected: only cards whose note metadata
is missing change grouping at all, so the damage must CONCENTRATE in users with a high NaN-note
rate. H2 makes no such prediction -- the teacher acts on every review, so its effect should be
roughly flat in that variable. Noise is flat too.

This is the same shape as prediction 3 of the interval pre-registration (concentration in same-day
rows), which is the check that turned "the means moved" into "the mechanism is real".

⚠ WHAT THIS CANNOT DO. It is an observational split of two existing runs, not a controlled arm, so
a positive result raises H1 from "possible" to "supported" -- it does not settle it. Only a run at
matched KD on the two databases does that. Stated here so the result is not over-read later.

Usage: nan_note_concentration.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import lmdb  # noqa: E402

from rwkv.data_processing import CARD_FEATURE_COLUMNS  # noqa: E402
from rwkv.prepare_batch import get_data  # noqa: E402

TEST_DB = "F:/rwkv_lmdb/test_db_5k_fixc"
TEST_DB_SIZE = 250_000_000_000
COL_NAN = CARD_FEATURE_COLUMNS.index("note_id_is_nan")

PAIRS = [
    ("ahead", "result/RWKV-iter53_muonlora.jsonl", "result/RWKV-fixc.jsonl"),
    ("imm", "result/RWKV-P-iter53_muonlora.jsonl", "result/RWKV-P-fixc.jsonl"),
]


def load(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["user"]] = r["metrics"]["LogLoss"]
    return out


def nan_note_rate_by_user():
    """Fraction of a user's REAL review rows whose card has no note metadata."""
    env = lmdb.open(TEST_DB, map_size=TEST_DB_SIZE, readonly=True, lock=False)
    rate = {}
    with env.begin(write=False) as txn:
        for user_id in range(5001, 7501):
            raw = txn.get(f"{user_id}_batches".encode())
            if raw is None:
                continue
            n_tot = n_nan = 0
            for b in json.loads(raw):
                d = get_data(txn, (user_id, b[0], b[1], b[2]), device="cpu")
                real = ~d.skips.numpy()
                cf = d.card_features.float().numpy()
                n_tot += int(real.sum())
                n_nan += int((cf[real, COL_NAN] > 0.5).sum())
            if n_tot:
                rate[user_id] = n_nan / n_tot
    return rate


def main():
    print("Discriminating H1 (note-pooling dose) from H2 (KD regime) by CONCENTRATION.")
    print("H1 predicts the fixc regression sits in users with many NaN-note cards; H2 predicts flat.")
    print()

    print("computing per-user NaN-note rate over eval users 5001-7500 ...", flush=True)
    rate = nan_note_rate_by_user()
    print("  %d users measured" % len(rate))
    if not rate:
        print("  *** no users measured -- FAILED TEST, not a pass")
        return 2

    vals = np.array(list(rate.values()))
    print("  NaN-note review share: median %.4f  mean %.4f  p90 %.4f  max %.4f"
          % (np.median(vals), vals.mean(), np.percentile(vals, 90), vals.max()))
    print()

    for mode, f_champ, f_cand in PAIRS:
        a, b = load(f_champ), load(f_cand)
        users = sorted(set(a) & set(b) & set(rate))
        r = np.array([rate[u] for u in users])
        d = np.array([b[u] - a[u] for u in users])  # positive = fixc WORSE

        order = np.argsort(r)
        q = np.array_split(order, 4)
        print("=== %s : fixc minus iter53 by NaN-note-rate quartile (positive = fixc worse) ===" % mode)
        print("  %-8s %8s %10s %12s" % ("quartile", "n", "mean rate", "mean delta"))
        for i, idx in enumerate(q):
            print("  Q%-7d %8d %10.4f %+12.6f" % (i + 1, len(idx), r[idx].mean(), d[idx].mean()))
        lo, hi = d[q[0]].mean(), d[q[3]].mean()
        print("  overall mean delta %+.6f" % d.mean())
        print("  top quartile minus bottom quartile: %+.6f" % (hi - lo))

        # Spearman, computed from ranks without scipy.
        rr = np.argsort(np.argsort(r)).astype(float)
        dd = np.argsort(np.argsort(d)).astype(float)
        rho = float(np.corrcoef(rr, dd)[0, 1])
        print("  Spearman rho(NaN-note rate, delta) = %+.4f" % rho)

        # Users with essentially no NaN-note cards isolate everything that is NOT H1.
        clean = d[r < 0.005]
        if clean.size:
            print("  users with <0.5%% NaN-note reviews: n=%d, mean delta %+.6f"
                  % (clean.size, clean.mean()))
            print("     ^ H1 predicts ~0 here. Anything clearly positive is NOT the note change.")
        print()

    print("READING IT: a strongly positive top-minus-bottom AND a near-zero clean-user delta")
    print("supports H1. A positive clean-user delta means something else is also moving, which")
    print("would be H2 or an unmodelled difference -- report both, do not pick the tidier one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
