"""v2: re-optimize FSRS-7 at EVERY checkpoint, on the prefix available then.

WHY v1 WAS WRONG (Andrew 2026-08-22): v1 read each user's parameters from
result/FSRS-7-short-secs.jsonl. Those are the FINAL vectors, fitted on all TimeSeriesSplit
folds but the last -- so they have already seen roughly 80% of the user's history,
including the future relative to almost every replay day. v1 therefore measured a
clairvoyant FSRS against a frozen RWKV. This version fits the parameters at each
checkpoint using only reviews up to that calendar day.

Two further changes he asked for, both in the same direction:
  * --recency is ON. It is the best FSRS-7 variant on the leaderboard (RMSE 0.3414 vs
    0.3437) and it is what a realistic deployment uses. The weighting is srs-benchmark's
    own (script._apply_recency_weighting), computed on the PREFIX length, which is what an
    optimizer run on day D would compute.
  * below MIN_TRAIN_ROWS the arm uses DEFAULT parameters instead of fitting. Not a fudge:
    it is what Anki does when a collection has too few reviews to optimize, and it makes
    the early checkpoints honest rather than unfittable.

THE v1 PARAMETERS RIDE ALONG AS A CONTROL. Each checkpoint also computes the workload the
STORED final vector would have produced, on the same cards, at the same day, with the same
mask. So "what did re-optimization change?" is answered by a within-run single-variable
A/B instead of by comparing two runs that also differ in their card mask and their day
sampling. Cost is one extra replay per checkpoint and no extra fit.

RWKV NEEDS NO RE-RUN, and that asymmetry is the point of the project: its weights are
frozen and user-independent, and the interval after review j depends only on reviews 1..j,
so v1's stored intervals already are what it would have produced at any checkpoint with
only the past in hand. This arm reuses out/rwkv_u*.parquet unchanged.

★ THE ACTIVE-CARD MASK IS PAST-ONLY, and the first version of this file got it wrong.
It asked whether the card's NEXT review found it in the review queue, which (a) peeks one
review into the future and (b) is far too restrictive -- median ZERO active cards for user
5530, so 37 of that user's 40 checkpoints were discarded. The mask now asks whether the
card's LAST review at or before D found it in the review queue and did not lapse
(state == 2 and rating > 1). That is a sufficient condition for "this is a review card
now", uses only information available at D, and lifts the median active count to 221-3908
across the users checked. It is derived from the shared table, so both arms are always
summed over exactly the same cards.

Usage:
  <srs-benchmark venv python> checkpoint_arm.py <table.parquet> <uid> <out.parquet> [step_days]
"""
import sys
import os
import json
import time
from pathlib import Path

SRSB = Path(r"C:\Users\Andrew\srs-benchmark")
REPO = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch")
sys.path.insert(0, str(SRSB))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# script.py builds its module-level `config` from sys.argv at import time, so the flags have
# to be in place BEFORE the import. This is what makes the fit below the benchmark's own
# --recency FSRS-7 fit rather than a reimplementation of it.
_ARGV = sys.argv[:]
sys.argv = ["script.py", "--algo", "FSRS-7", "--short", "--secs", "--recency",
            "--processes", "1"]

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

_CWD = os.getcwd()
os.chdir(SRSB)          # script.py resolves pretrain/ and result/ relative to cwd
import script           # noqa: E402
from data_loader import UserDataLoader  # noqa: E402
from models.fsrs_v7 import FSRS7  # noqa: E402
os.chdir(_CWD)

script.config.device = torch.device("cpu")

from fsrs_arm import intervals_at_dr, replay_states, DR_LEVELS  # noqa: E402

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")
# the --recency stored vectors, to match the refit's own flag
FINAL_PARAMS = SRSB / "result" / "FSRS-7-short-secs-recency.jsonl"
# Anki refuses to optimize a collection with too few reviews and runs on defaults; 400 is
# the scale of its own threshold.
MIN_TRAIN_ROWS = 400
MIN_ACTIVE = 20
FLOOR_DAYS = 1.0


def load_final_w(uid):
    with open(FINAL_PARAMS, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("user") == uid:
                return [float(x) for x in rec["parameters"]["0"]]
    return None


def last_review_per_card(day_off, card, day):
    """Index of each card's last review at or before `day`. Rows are in review_th order."""
    idx = np.nonzero(day_off <= day)[0]
    if not len(idx):
        return idx
    c = card[idx]
    _, last_pos = np.unique(c[::-1], return_index=True)
    return idx[len(idx) - 1 - last_pos]


def workload(model, prefix, table, act, review_th_pos):
    """sum(1/interval) over the active cards, per DR level, for one parameter vector."""
    tb, st = replay_states(model, prefix)
    pos = pd.Series(np.arange(len(tb)), index=tb["review_th"].to_numpy())
    sel = pos.loc[review_th_pos].to_numpy()
    ivl, _, _ = intervals_at_dr(model, st[sel], DR_LEVELS)
    return (1.0 / np.maximum(ivl, FLOOR_DAYS)).sum(axis=0), ivl


def main():
    table_path = Path(_ARGV[1])
    uid = int(_ARGV[2])
    out_path = Path(_ARGV[3])
    step = int(_ARGV[4]) if len(_ARGV) > 4 else 50

    table = pd.read_parquet(table_path)
    raw = pd.read_parquet(DATA / "revlogs" / ("user_id=%d" % uid))
    assert len(raw) == len(table), "canonical table and raw revlogs disagree on row count"

    os.chdir(SRSB)
    ds = UserDataLoader(script.config).load_user_data(uid)
    os.chdir(_CWD)
    ds = ds.sort_values("review_th", kind="stable").reset_index(drop=True)

    rwkv = pd.read_parquet(REPO / "scratchpad" / "workload" / "out" / ("rwkv_u%d.parquet" % uid))
    assert len(rwkv) == len(table), "RWKV arm and canonical table disagree on row count"
    assert (rwkv["review_th"].to_numpy() == table["review_th"].to_numpy()).all()
    rwkv_ivl = rwkv[["ivl_%d" % round(dr * 100) for dr in DR_LEVELS]].to_numpy(dtype=np.float64)

    day_off = table["day_offset"].to_numpy()
    card = table["card_id"].to_numpy()
    rth = table["review_th"].to_numpy()
    # past-only review-queue test, evaluated on whichever review is the card's latest
    in_queue = (raw["state"].to_numpy() == 2) & (raw["rating"].to_numpy() > 1)

    # ★ THE ALIVE SUBSET, and why it is recorded separately rather than used as the mask.
    # A past-only mask cannot know that a card is never touched again, so the active set is
    # mostly ABANDONED cards: at user 5100 day 600, only 22 of 187 active cards are ever
    # reviewed again (the user keeps adding new cards and dropping old ones). Both arms
    # carry the same phantom cards, so the RATIO stays a fair comparison -- but it is not
    # obviously the same ratio you would get on cards that are really in rotation, and the
    # absolute anchor is impossible on them, because a card with no next review has no
    # actual interval to compare against. So: the headline mask stays past-only, and these
    # extra sums make "does it matter?" a measurement instead of an argument.
    _ed = raw["elapsed_days"].to_numpy(dtype=np.float64)
    _o = np.argsort(card, kind="stable")
    _cs = card[_o]
    _tmp = np.full(len(table), np.nan)
    _tmp[:-1] = np.where(_cs[:-1] == _cs[1:], _ed[_o][1:], np.nan)
    gap_days = np.full(len(table), np.nan)
    gap_days[_o] = _tmp        # gap to that card's NEXT review, in days; nan = never again

    w_final = load_final_w(uid)
    lo, hi = int(day_off.min()), int(day_off.max())
    rows = []
    t0 = time.perf_counter()
    for D in range(lo + step, hi + 1, step):
        last = last_review_per_card(day_off, card, D)
        act = last[in_queue[last]]
        if len(act) < MIN_ACTIVE:
            continue
        train_df = ds[ds["day_offset"] <= D]
        used_default = len(train_df) < MIN_TRAIN_ROWS
        if used_default:
            w = list(FSRS7.init_w)
        else:
            os.chdir(SRSB)
            try:
                w = script._fit_trainable_weights(script._apply_recency_weighting(train_df))
            finally:
                os.chdir(_CWD)
            w = [float(x) for x in w]

        prefix = table[day_off <= D]
        sel_rth = rth[act]
        alive = np.isfinite(gap_days[act])          # boolean over the ACTIVE rows
        m_refit = FSRS7(script.config, w=w).to(script.config.device).eval()
        _, ivl_f = workload(m_refit, prefix, table, act, sel_rth)
        rate_f = 1.0 / np.maximum(ivl_f, FLOOR_DAYS)
        rec = {"user": uid, "day": D, "n_active": int(len(act)),
               "n_alive": int(alive.sum()),
               "n_train": int(len(train_df)), "used_default": bool(used_default)}
        rate_r = 1.0 / np.maximum(rwkv_ivl[act], FLOOR_DAYS)
        rate_v1 = None
        if w_final is not None:
            m_final = FSRS7(script.config, w=w_final).to(script.config.device).eval()
            _, ivl_v1 = workload(m_final, prefix, table, act, sel_rth)
            rate_v1 = 1.0 / np.maximum(ivl_v1, FLOOR_DAYS)
        for k, dr in enumerate(DR_LEVELS):
            p = round(dr * 100)
            rec["wf_%d" % p] = float(rate_f[:, k].sum())
            rec["wr_%d" % p] = float(rate_r[:, k].sum())
            rec["wfa_%d" % p] = float(rate_f[alive, k].sum())
            rec["wra_%d" % p] = float(rate_r[alive, k].sum())
            if rate_v1 is not None:
                rec["wv1_%d" % p] = float(rate_v1[:, k].sum())
                rec["wv1a_%d" % p] = float(rate_v1[alive, k].sum())
        # the load the user really carried on exactly those alive cards
        g = gap_days[act][alive]
        rec["w_actual"] = float((1.0 / np.maximum(g, FLOOR_DAYS)).sum()) if alive.any() else 0.0
        rows.append(rec)

    el = time.perf_counter() - t0
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    meta = {"user": uid, "step_days": step, "n_checkpoints": len(out),
            "n_default_param_checkpoints": int(out["used_default"].sum()) if len(out) else 0,
            "min_train_rows": MIN_TRAIN_ROWS, "min_active": MIN_ACTIVE, "recency": True,
            "floor_days": FLOOR_DAYS, "has_v1_control": w_final is not None,
            "seconds": el, "dr_levels": DR_LEVELS}
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if len(out):
        r = out["wf_90"].sum() / out["wr_90"].sum()
        r1 = (out["wv1_90"].sum() / out["wr_90"].sum()) if w_final is not None else float("nan")
        print("u%-6d %3d ckpts (%d default) %5.1f min  median active %5d  F/R@90 refit %.3f  final %.3f"
              % (uid, len(out), meta["n_default_param_checkpoints"], el / 60,
                 int(out["n_active"].median()), r, r1))
    else:
        print("u%-6d no usable checkpoints" % uid)


if __name__ == "__main__":
    main()
