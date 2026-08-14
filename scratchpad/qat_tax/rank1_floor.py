#!/usr/bin/env python
"""The EXACT rank-1 truncation floor of a WKV state corpus -- the decisive check for the rank-1
regularizer (iter qtaxf_r1reg).

WHY THIS EXISTS. The deploy quantizer truncates each per-head KxK WKV state to rank 1, and that
truncation is the single largest term in the reconstruction ladder (53% card / 39% note). It is
STRUCTURALLY frozen -- no number of codebook bits touches it -- so the only lever on it is to TRAIN
the model to emit states that are more nearly rank-1. `RWKV_QAT_RANK1_REG` does that, and Andrew's
objection is that the model would already be doing it if it paid ("if making states more rank-1 could
lower log loss, the model would learn to do that anyway").

The run alone cannot distinguish the two ways it can come back null:
  * the regularizer never moved the states  -> lever untested, and a bigger lambda is indicated;
  * the states moved and the loss did not   -> **Andrew is right, empirically**, rank-1-ness is
    achievable and worthless, and the whole sub-family closes.
This script measures the quantity that separates them. Run it on the CONTROL checkpoint and the
REGULARIZED one and compare; the training-loss penalty term measures the PROXY (k/v alignment), which
is not the same thing -- the proxy ignores the decay weighting, its known blunt edge.

METRIC. Per head-state A (KxK), the exact rank-1 error is fixed by the singular values:

    ||A - A_1||_F / ||A||_F = sqrt(1 - sigma_1^2 / sum_i sigma_i^2)

so no explicit truncation is needed -- one SVD per state. Reported as the MEAN over states, matching
how the ladder's 0.4353 (card) / 0.3049 (note) were produced. `participation` is the same
concentration measure the regularizer penalizes, computed HERE on the state itself rather than on the
kernel inputs, which is exactly the proxy-vs-truth gap the check is about.

⚠ Reconstruction error is NOT a proxy for logloss -- three mispredictions in one week, in both
directions (see research_5k_notes.md). This answers "did the states move?", never "is it better?".

Usage:
    python scratchpad/qat_tax/rank1_floor.py "scratchpad/qat_tax/corpus/wkv_*_card.txt" [--h 5] [--k 16]
    python scratchpad/qat_tax/rank1_floor.py <glob> --label r1reg --json out.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pq_train import load_states  # noqa: E402


def spectrum_stats(states):
    """states: [N, H, K, K]. Returns per-(state,head) arrays of rank-1 error and participation."""
    n, h, k, _ = states.shape
    flat = states.reshape(n * h, k, k)
    sv = np.linalg.svd(flat, compute_uv=False)          # [N*H, K], descending
    energy = (sv ** 2).sum(axis=1)
    ok = np.isfinite(energy) & (energy > 1e-30)
    sv, energy = sv[ok], energy[ok]
    top = sv[:, 0] ** 2
    rank1_err = np.sqrt(np.maximum(0.0, 1.0 - top / energy))
    # participation ratio of the SPECTRUM: 1 = rank-1, 1/r for r equal singular values. This is the
    # same functional the regularizer uses, but applied to the state rather than to k/v alignment.
    part = top ** 0 * (top / energy)                     # sigma_1^2 / sum sigma_i^2
    conc = ((sv ** 2) ** 2).sum(axis=1) / (energy ** 2)  # ||A^T A||_F^2 / ||A||_F^4
    return rank1_err, part, conc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", help="glob(s) of --dump-corpus output files")
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    files = sorted({f for g in a.corpus for f in glob.glob(g)})
    if not files:
        raise SystemExit(f"no corpus files matched {a.corpus}")
    states = load_states(files, a.h, a.k)
    if states.size == 0:
        raise SystemExit(f"{len(files)} files matched but yielded 0 states (wrong --h/--k?)")

    err, part, conc = spectrum_stats(states)
    out = {
        "label": a.label,
        "files": len(files),
        "states": int(states.shape[0]),
        "head_states": int(err.size),
        "rank1_err_mean": float(err.mean()),
        "rank1_err_median": float(np.median(err)),
        "rank1_err_p90": float(np.percentile(err, 90)),
        "energy_in_rank1_mean": float(part.mean()),
        "spectral_concentration_mean": float(conc.mean()),
    }
    tag = f" [{a.label}]" if a.label else ""
    print(f"{len(files)} files, {states.shape[0]:,} states, {err.size:,} head-states"
          f" (H={a.h} K={a.k}){tag}")
    print(f"  EXACT RANK-1 ERROR  mean {out['rank1_err_mean']:.4f}"
          f"  median {out['rank1_err_median']:.4f}  p90 {out['rank1_err_p90']:.4f}")
    print(f"  energy kept by rank-1     {out['energy_in_rank1_mean']:.4f}")
    print(f"  spectral concentration    {out['spectral_concentration_mean']:.4f}   (1.0 = rank-1)")
    print("  reference (iter45 champion, frozen recipe): card 0.4353 / note 0.3049")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
