"""FSRS-7 arm: replay a user's real history and record the interval FSRS-7 WOULD assign
after every review, at each desired-retention level.

RUNS IN srs-benchmark's VENV, with srs-benchmark on sys.path, so the recurrence and the
forgetting curve are the benchmark's OWN code (models.fsrs_v7.FSRS7). Nothing about
FSRS-7 is reimplemented here -- only the interval inversion, which the benchmark has no
non-differentiable version of.

PARAMETERS ARE NOT RE-OPTIMIZED. srs-benchmark already stores the per-user optimized
34-param vector for every user in result/FSRS-7-*.jsonl (10,000 users, written
2026-06-26). Re-running the optimizer would produce the same thing at enormous cost.
The vector actually used is copied into the output meta so the run stays reproducible
even if an in-flight re-benchmark overwrites the source file.

Usage:
  <srs-benchmark venv python> fsrs_arm.py <table.parquet> <user_id> <out.parquet>
"""
import sys
import os
import json
import math
from pathlib import Path

SRSB = Path(r"C:\Users\Andrew\srs-benchmark")
sys.path.insert(0, str(SRSB))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

import config as _cfgmod

assert Path(_cfgmod.__file__).parent == SRSB, (
    "wrong config.py on sys.path (%s) -- this must be srs-benchmark's, not the "
    "rwkv repo's vendored copy" % _cfgmod.__file__
)
from config import Config, create_parser
from models.fsrs_v7 import FSRS7

# The headline FSRS-7 row in srs-benchmark's README:
#   python script.py --algo FSRS-7 --short --secs --equalize_test_with_non_secs
# equalize_test_with_non_secs changes only WHICH ROWS ARE SCORED (create_features.py
# intersects the secs frame with the non-secs one); the trained features are the secs
# ones either way, so these parameters are the ordinary --short --secs parameters.
#
# "sched" is the sched_penalties variant. It matters HERE more than it does in the
# benchmark: its two penalties exist to stop FSRS-7 asking for sub-second intervals at
# high desired retention, which is exactly the regime this study measures, and
# sum(1/interval) is dominated by the shortest intervals. Accuracy is within noise of the
# plain variant (README: RMSE 0.3438 vs 0.3437), so it is the fairer scheduler-side arm.
#
# THE NON-EQUALIZED FILES ARE THE RIGHT ONES HERE, and the two are NOT interchangeable:
# the parameter vectors genuinely differ (median max-abs difference 0.094 over 200 users,
# max 1.28), because equalization changes which rows train as well as which score. The
# canonical replay table is the full raw stream, which is what the plain --short --secs
# pipeline fits; and its recorded per-user LogLoss is therefore the number this replay can
# be validated against. (Cross-check on user 5100: replay 0.4078 over 4905 rows vs the
# benchmark's recorded 0.4084 over 4085 -- the model and the parameters are reproduced.)
VARIANTS = {
    "plain": "FSRS-7-short-secs.jsonl",
    "sched": "FSRS-7-sched_penalties-short-secs.jsonl",
    "default": "FSRS-7-default-short-secs.jsonl",
    "recency": "FSRS-7-short-secs-recency.jsonl",
}
PARAM_FILE = SRSB / "result" / VARIANTS["plain"]

DR_LEVELS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
SECONDS_PER_DAY = 86400.0
MIN_T = 1.0 / SECONDS_PER_DAY   # 1 second, in days
MAX_T = 36500.0                 # 100 years, in days
N_BISECT = 60


def make_config():
    """The Config the headline FSRS-7 run uses. Built through the real parser so every
    default (s_min, init_s_max, ...) is the benchmark's, not a guess."""
    parser = create_parser()
    args = parser.parse_args([
        "--algo", "FSRS-7", "--short", "--secs",
        "--equalize_test_with_non_secs", "--processes", "1",
    ])
    cfg = Config(args)
    cfg.device = torch.device("cpu")
    return cfg


def load_params(user_id):
    """The user's optimized 34-param vector, plus the LogLoss the benchmark recorded for
    it. The recorded value is carried through so every user gets a validation line: this
    replay should land near it, on a row set that is close but not identical (the
    benchmark's pipeline drops rows this one keeps)."""
    with open(PARAM_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("user") == user_id:
                p = rec["parameters"]
                # partitions="none" -> a single partition keyed "0"
                assert list(p.keys()) == ["0"], "unexpected partition keys %s" % list(p)
                w = p["0"]
                assert len(w) == 34, "FSRS-7 must have 34 parameters, got %d" % len(w)
                return w, float(rec["metrics"]["LogLoss"]), int(rec["size"])
    raise KeyError("user %d not in %s" % (user_id, PARAM_FILE.name))


def replay_states(model, table):
    """State after EVERY review, via the model's own recurrence.

    Cards are independent given their own (delta_t, rating) prefix, so the whole user is
    one padded batch: inputs (max_len, n_cards, 2) -> FSRS7.forward gives the state after
    each step. Padded tail positions are stepped too but never read; they cannot feed back
    into earlier positions, so their values are irrelevant.
    """
    tb = table.sort_values(["card_id", "review_th"], kind="stable")
    card_ids = tb["card_id"].to_numpy()
    uniq, first_idx, lengths = np.unique(card_ids, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    uniq, lengths = uniq[order], lengths[order]
    max_len, n_cards = int(lengths.max()), len(uniq)

    delta_t = np.maximum(
        0.0, tb["elapsed_seconds"].to_numpy(dtype=np.float64)) / SECONDS_PER_DAY
    rating = tb["rating"].to_numpy(dtype=np.float64)

    # pad with (1 day, Good): never read, and keeps the padded state finite
    seq = np.zeros((max_len, n_cards, 2), dtype=np.float32)
    seq[:, :, 0] = 1.0
    seq[:, :, 1] = 3.0
    slot = {c: k for k, c in enumerate(uniq)}
    slots = np.array([slot[c] for c in card_ids])
    within = np.concatenate([np.arange(L) for L in lengths])
    # tb is sorted by (card_id, review_th); `within` is built from the per-card lengths in
    # first-appearance order, so block k of `within` lines up with card k. Asserted, not
    # assumed -- a mismatch here would silently scramble which review got which state.
    assert (np.bincount(slots, minlength=n_cards) == lengths).all()
    assert len(within) == len(slots)
    seq[within, slots, 0] = delta_t
    seq[within, slots, 1] = rating

    with torch.no_grad():
        outputs, _ = model.forward(torch.from_numpy(seq))   # (max_len, n_cards, 3)
    st = outputs[within, slots]                             # (n_reviews, 3) in tb order
    return tb, st


def intervals_at_dr(model, st, dr_levels):
    """t such that R(t, S, S_short, D) = dr, by bisection in log t.

    Bisection rather than the benchmark's Newton solver (fsrs_v7_interval_penalty) because
    that one exists to be DIFFERENTIABLE and runs a fixed 7 steps from a log(s) start;
    here there is no gradient to carry and a guaranteed bracket matters more than speed.
    The dual-trace curve is strictly decreasing in t (both components decrease, both
    mixture weights are positive), so the bracket is valid by construction.

    Returns (ivl_days (M,K), hit_lo (M,K), hit_hi (M,K)). The two flags say the target was
    NOT reachable inside [1 second, 100 years]: hit_hi means the card is still above the
    target at 100 years, hit_lo that it is already below it after 1 second. Both are
    reported rather than silently clamped -- 1/interval explodes at the low end.
    """
    S = st[:, 0].double().unsqueeze(1)
    Ss = st[:, 1].double().unsqueeze(1)
    D = st[:, 2].double().unsqueeze(1)
    tgt = torch.tensor(dr_levels, dtype=torch.float64).unsqueeze(0)

    w_backup = model.w
    model.w = torch.nn.Parameter(model.w.data.double(), requires_grad=False)
    try:
        def R(t):
            return model.forgetting_curve(t, S, Ss, D)

        shape = torch.broadcast_shapes(S.shape, tgt.shape)
        lo = torch.full(shape, math.log(MIN_T), dtype=torch.float64)
        hi = torch.full(shape, math.log(MAX_T), dtype=torch.float64)
        hit_lo = (R(torch.exp(lo)) < tgt).expand(shape).clone()
        hit_hi = (R(torch.exp(hi)) > tgt).expand(shape).clone()
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            above = (R(torch.exp(mid)) > tgt).expand(shape)
            lo = torch.where(above, mid, lo)
            hi = torch.where(above, hi, mid)
        ivl = torch.exp(0.5 * (lo + hi))
    finally:
        model.w = w_backup
    return ivl.numpy(), hit_lo.numpy(), hit_hi.numpy()


def ahead_predictions(model, tb, st):
    """The scheduler's OWN curve evaluated at the interval that actually happened, plus
    the outcome -- so the arm can be calibration-checked on the very rows it schedules.

    WHY THIS IS NEEDED FOR THE WORKLOAD RESULT, not just as a health check. The workload
    ratio compares the two algorithms at the same NOMINAL desired retention. That is only
    a fair comparison if both algorithms deliver the retention they claim: an overconfident
    model asks for longer intervals, so it "wins" on workload while quietly under-shooting
    the target. Comparing mean(predicted) with mean(observed) on identical rows is what
    turns "FSRS asks for longer intervals" into either "FSRS is more efficient" or "FSRS is
    overconfident here".

    tb is sorted by (card_id, review_th), so the next review of the same card is the next
    row whenever card_id is unchanged. The last review of each card has no outcome.
    """
    card = tb["card_id"].to_numpy()
    dt = np.maximum(0.0, tb["elapsed_seconds"].to_numpy(dtype=np.float64)) / SECONDS_PER_DAY
    rating = tb["rating"].to_numpy()
    has_next = np.zeros(len(tb), dtype=bool)
    has_next[:-1] = card[:-1] == card[1:]
    next_dt = np.zeros(len(tb), dtype=np.float64)
    next_dt[:-1] = dt[1:]
    next_y = np.zeros(len(tb), dtype=np.float64)
    next_y[:-1] = (rating[1:] > 1).astype(np.float64)

    w_backup = model.w
    model.w = torch.nn.Parameter(model.w.data.double(), requires_grad=False)
    try:
        pred = model.forgetting_curve(
            torch.from_numpy(next_dt), st[:, 0].double(), st[:, 1].double(),
            st[:, 2].double()).numpy()
    finally:
        model.w = w_backup
    return pred, next_y, has_next


def main():
    table_path, user_id, out_path = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    variant = sys.argv[4] if len(sys.argv) > 4 else "plain"
    global PARAM_FILE
    PARAM_FILE = SRSB / "result" / VARIANTS[variant]
    table = pd.read_parquet(table_path)
    cfg = make_config()
    w, bench_logloss, bench_size = load_params(user_id)
    model = FSRS7(cfg, w=w).to(cfg.device)
    model.eval()

    tb, st = replay_states(model, table)
    ivl, hit_lo, hit_hi = intervals_at_dr(model, st, DR_LEVELS)
    pred, y, has_next = ahead_predictions(model, tb, st)

    out = pd.DataFrame({
        "card_id": tb["card_id"].to_numpy(),
        "review_th": tb["review_th"].to_numpy(),
        "day_offset": tb["day_offset"].to_numpy(),
        "rating": tb["rating"].to_numpy(),
        "pred": pred,
        "y": y,
        "has_next": has_next,
    })
    for k, dr in enumerate(DR_LEVELS):
        out["ivl_%d" % round(dr * 100)] = ivl[:, k]
    out = out.sort_values("review_th", kind="stable").reset_index(drop=True)
    m = out["has_next"].to_numpy()
    p = np.clip(out["pred"].to_numpy()[m], 1e-6, 1 - 1e-6)
    yy = out["y"].to_numpy()[m]
    logloss = float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    meta = {
        "user": user_id,
        "n_reviews": int(len(out)),
        "n_cards": int(out["card_id"].nunique()),
        "param_file": PARAM_FILE.name,
        "variant": variant,
        "w": w,
        "dr_levels": DR_LEVELS,
        "n_scored": int(m.sum()),
        "logloss": logloss,
        "mean_pred": float(p.mean()),
        "mean_actual": float(yy.mean()),
        "calibration_bias": float(p.mean() - yy.mean()),
        "benchmark_logloss": bench_logloss,
        "benchmark_size": bench_size,
        "hit_lo_frac": [float(hit_lo[:, k].mean()) for k in range(len(DR_LEVELS))],
        "hit_hi_frac": [float(hit_hi[:, k].mean()) for k in range(len(DR_LEVELS))],
        "s_min": cfg.s_min,
    }
    with open(str(out_path) + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("FSRS user %d: %d reviews -> %s" % (user_id, len(out), out_path))
    print("  median ivl (days) by DR: " + "  ".join(
        "%d%%=%.3f" % (round(dr * 100), float(np.median(ivl[:, k])))
        for k, dr in enumerate(DR_LEVELS)))
    print("  unreachable-low frac:    " + "  ".join(
        "%d%%=%.4f" % (round(dr * 100), meta["hit_lo_frac"][k])
        for k, dr in enumerate(DR_LEVELS)))
    print("  ahead logloss %.4f on %d rows; mean pred %.4f vs actual %.4f (bias %+.4f)"
          % (logloss, int(m.sum()), p.mean(), yy.mean(), p.mean() - yy.mean()))
    print("  VALIDATION: benchmark recorded %.4f on %d rows (delta %+.4f)"
          % (bench_logloss, bench_size, logloss - bench_logloss))


if __name__ == "__main__":
    main()
