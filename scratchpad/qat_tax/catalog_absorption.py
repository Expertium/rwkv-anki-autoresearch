#!/usr/bin/env python
"""Has CATALOG LEARNING already absorbed what a better INITIAL catalog would buy? (CPU, minutes)

THE DECISION THIS MAKES. With the rank-1 term closed (iter 47) the remaining QAT tax lives in the
codebook and norm terms, and the norm levers are closed on mechanism. The obvious surviving candidate
is the "free axis" recorded in CLAUDE.md: at bits=12 the ORACLE reached 0.2044 against a REFIT of
0.3224, so CORPUS SIZE limits the fit and more users cost no deploy bytes.

But that gap was measured with FROZEN catalogs. Since then the catalogs LEARN during the run, and the
pattern that closed four norm levers is that **a learnable component silently absorbs the levers
beside it**. So before spending a GPU run on a better-initialised catalog, ask the cheap version:

    how much of the initial-catalog deficit did LEARNING already remove,
    and how much headroom is left to the oracle?

  INITIAL   reference/pq_cb_wkv_c80_b10.txt -- what the run started from
  LEARNED   <run>_wkvcb_10935.txt           -- what the run ended with
  ORACLE    k-means fitted on the TRAIN users, scored on the HELD-OUT user

**If LEARNED is at or near ORACLE, a better init cannot help and the axis is CLOSED** -- learning
already reaches the achievable floor at 1024 centroids. If a wide gap remains, the axis is open and a
better-initialised run is justified.

⚠ HONEST SPLIT. States from one user's cards are correlated, so a random vector split flatters the
oracle: a held-out vector usually has a near-twin in training. Hold out a whole USER, matching
wkv_cb_staleness.py's --holdout-user reasoning, and score all three catalogs on the SAME held-out
vectors so the comparison is paired.
⚠ Reconstruction error is NOT a logloss proxy -- four mispredictions this week, in both directions.
This answers "is there headroom in the catalog", never "what would it be worth".

Usage:
  python scratchpad/qat_tax/catalog_absorption.py --holdout-user 136 \
      --learned scratchpad/iter45_kddecay/qtaxd_cblearn_d_wkvcb_10935.txt \
      "scratchpad/qat_tax/corpus_ctrl_final/wkv_*_card.txt" \
      "scratchpad/qat_tax/corpus_ctrl_final/wkv_*_note.txt"
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wkv_cb_staleness import collect_joint, encode_err, fit, load_joint_cb  # noqa: E402
from pq_train import load_states  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--holdout-user", required=True)
    ap.add_argument("--learned", required=True)
    ap.add_argument("--initial", default="reference/pq_cb_wkv_c80_b10.txt")
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3, help="oracle k-means restarts; report the best")
    a = ap.parse_args()

    files = sorted({f for g in a.corpus for f in glob.glob(g)})
    hold_f = [f for f in files if f"_{a.holdout_user}_" in os.path.basename(f)]
    train_f = [f for f in files if f not in hold_f]
    if not hold_f or not train_f:
        raise SystemExit(f"holdout {a.holdout_user}: {len(hold_f)} hold / {len(train_f)} train files")
    print(f"holdout user {a.holdout_user}: {len(hold_f)} files; train: {len(train_f)} files")

    hold = collect_joint(load_states(hold_f, a.h, a.k), a.k)
    train = collect_joint(load_states(train_f, a.h, a.k), a.k)
    print(f"  joint (u,v) vectors: hold {len(hold):,}  train {len(train):,}  dim {hold.shape[1]}")

    cb_i, bits_i, _ = load_joint_cb(a.initial)
    cb_l, bits_l, _ = load_joint_cb(a.learned)
    ncent = cb_i.shape[0]
    assert cb_l.shape == cb_i.shape, f"shape mismatch {cb_l.shape} vs {cb_i.shape}"

    e_init = encode_err(hold, cb_i)
    e_learn = encode_err(hold, cb_l)
    best = None
    for s in range(a.seeds):
        e = encode_err(hold, fit(train, ncent, seed=s))
        best = e if best is None else min(best, e)
    e_oracle = best

    # A catalog of RANDOM directions, as the "is this better than nothing" control that caught the
    # stale q72u catalog being worse than random. Same count, unit-normalised like the data.
    rng = np.random.default_rng(0)
    rnd = rng.normal(size=(ncent, hold.shape[1])).astype(np.float32)
    rnd /= np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
    rnd *= np.median(np.linalg.norm(hold, axis=1))
    e_rand = encode_err(hold, rnd)

    print(f"\nheld-out mean relative L2 ({ncent} centroids, {bits_i} bits):")
    print(f"  RANDOM   {e_rand:.4f}   (control; 1.0 = encode-everything-to-zero)")
    print(f"  INITIAL  {e_init:.4f}   {os.path.basename(a.initial)}")
    print(f"  LEARNED  {e_learn:.4f}   {os.path.basename(a.learned)}")
    print(f"  ORACLE   {e_oracle:.4f}   k-means on the train users, best of {a.seeds} seeds")

    gained = e_init - e_learn
    headroom = e_learn - e_oracle
    total = e_init - e_oracle
    print(f"\n  learning removed   {gained:+.4f}")
    print(f"  headroom to oracle {headroom:+.4f}")
    if total > 1e-9:
        print(f"  => learning captured {gained / total:.0%} of the initial-to-oracle gap")
    print()
    # ⚠ NO RUN RECOMMENDATION IS MADE FROM THESE NUMBERS, and the first version of this script was
    # WRONG to make one. Measured 2026-08-15: the LEARNED catalog reconstructs WORSE than the initial
    # one on BOTH corpora (0.513 vs 0.334 on champion states, 0.907 vs 0.877 on QAT-final states) and
    # still delivered a 45%/44% QAT-tax reduction. The reason is structural: centroids are trained by
    # dL/dcentroid on the TASK loss, never on reconstruction error, so they are free to move somewhere
    # that serves the task and reconstructs worse. Reconstruction therefore cannot even RANK two
    # catalogs, let alone price one -- which makes "oracle 0.2044 vs refit 0.3224, so more users is a
    # free win" an argument with no demonstrated link to logloss.
    print("HEADROOM (reconstruction only):")
    print(f"  learning moved reconstruction by {gained:+.4f}; oracle is {headroom:+.4f} away.")
    print("⚠ DO NOT turn this into a run decision. A learned catalog optimises the TASK loss, not")
    print("  reconstruction, and is measured here to reconstruct WORSE than its own starting point")
    print("  while cutting the QAT tax ~45%. Reconstruction cannot rank catalogs. To price a better")
    print("  initial catalog you need an A/B on LOGLOSS, not this number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
