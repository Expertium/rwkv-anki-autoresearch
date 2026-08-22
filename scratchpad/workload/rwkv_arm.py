"""RWKV-Curve arm: replay a user's real history through the FROZEN champion and record
the interval it WOULD assign after every review, at each desired-retention level.

THE DEPLOY CONTRACT IS FOLLOWED, not approximated (CLAUDE.md, "THE DEPLOY CONTRACT"):
  1. the most recent review's duration is zeroed  -> button_heads sets RWKV_PROBE_DUR
  2. PAVA rectification is applied                -> button_curves
  3. no piecewise ahead correction                -> RWKV_NO_AHEAD_RESIDUAL=1
so the number this produces is what the Anki-side scheduler would actually compute, not
what the training objective sees.

WHY FIVE FORWARD PASSES PER REVIEW. PAVA pools ACROSS the four counterfactual buttons at
each horizon, so the rectified curve for the button the user actually pressed cannot be
obtained from that button alone -- all four heads are needed (4 passes, state NOT
advanced). The real row is then fed through review() with its REAL duration to advance
the state (1 pass). Measured 20.5 reviews/s on one thread.

INTERVAL INVERSION. The heads do not depend on the horizon, so the whole rectified curve
over a log-spaced t grid costs one button_curves call; the seven DR levels are then read
off by linear interpolation in (log t, R). That is one curve evaluation instead of one
bisection per DR.

Usage: .venv/Scripts/python.exe scratchpad/workload/rwkv_arm.py <table.parquet> <uid> <out.parquet>
"""
import sys
import os
import json
import time

sys.path.insert(0, os.getcwd())
from scratchpad.workload.env_champ import apply, CHAMPION_CKPT  # noqa: E402

APPLIED_ENV = apply()

from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from rwkv.data_processing import get_rwkv_data  # noqa: E402
from rwkv.run_as_rnn import RNNProcess  # noqa: E402
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")
DR_LEVELS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
SECONDS_PER_DAY = 86400.0
N_GRID = 441           # log-spaced over [1 s, exp(s_max)] -> ~0.05 in log t before interp
ID_ENCODING_SEED = 1234


def build_grid(s_max):
    """Log-spaced horizons in SECONDS, from 1 s to exp(s_max) (~113 years, the same
    ceiling button_intervals uses). The curve clamps t at 1 s internally, so 1 s is the
    smallest horizon that means anything."""
    return torch.exp(torch.linspace(0.0, float(s_max), N_GRID, dtype=torch.float32))


def invert(curve_np, log_t, targets):
    """Read t(DR) off one decreasing curve sampled on a log-t grid.

    Returns (t_seconds (K,), hit_lo (K,), hit_hi (K,), n_violations).

    The curve SHOULD be decreasing in t: every raw button curve is monotone by
    construction (gru_forgetting_curve uses exp(-d*log1p(t/S)) with d > 0), but PAVA
    pools a possibly DIFFERENT set of buttons at each horizon, so monotonicity of the
    RECTIFIED curve is not guaranteed by that argument. A running minimum is applied and
    the violations are counted rather than assumed away.
    """
    mono = np.minimum.accumulate(curve_np)
    n_viol = int((curve_np - mono > 1e-9).sum())
    out = np.empty(len(targets), dtype=np.float64)
    hit_lo = np.zeros(len(targets), dtype=bool)
    hit_hi = np.zeros(len(targets), dtype=bool)
    for k, dr in enumerate(targets):
        if mono[0] < dr:                     # below target already at 1 second
            hit_lo[k] = True
            out[k] = float(np.exp(log_t[0]))
            continue
        if mono[-1] > dr:                    # still above target at the ceiling
            hit_hi[k] = True
            out[k] = float(np.exp(log_t[-1]))
            continue
        j = int(np.searchsorted(-mono, -dr))  # first index with mono[j] < dr
        j = max(1, min(j, len(mono) - 1))
        y0, y1 = mono[j - 1], mono[j]
        x0, x1 = log_t[j - 1], log_t[j]
        out[k] = float(np.exp(x0 if y0 == y1 else x0 + (dr - y0) * (x1 - x0) / (y1 - y0)))
    return out, hit_lo, hit_hi, n_viol


def main():
    table_path, user_id, out_path = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    table = pd.read_parquet(table_path)

    df = get_rwkv_data(DATA, user_id)
    df = df.sort_values("review_th", kind="stable").reset_index(drop=True)
    # The two arms MUST replay the same rows. Assert it instead of hoping: this is the
    # single failure that would make the whole workload ratio meaningless while every
    # individual number still looked sane.
    assert len(df) == len(table), "row count %d vs canonical %d" % (len(df), len(table))
    for col in ("card_id", "review_th", "day_offset", "rating"):
        assert (df[col].to_numpy() == table[col].to_numpy()).all(), "column %s differs" % col

    proc = RNNProcess(CHAMPION_CKPT, "cpu", torch.float32, DEFAULT_ANKI_RWKV_CONFIG)
    rnn = proc.rnn
    assert rnn.gru_on, "champion uses the GRU curve head"
    assert hasattr(rnn, "pava_theta"), "champion carries the learned PAVA rectifier"

    torch.manual_seed(ID_ENCODING_SEED)   # the id encodings are randint draws
    t_grid = build_grid(rnn.s_max)
    log_t = np.log(t_grid.numpy().astype(np.float64))

    n = len(df)
    ivl = np.empty((n, len(DR_LEVELS)), dtype=np.float64)
    hit_lo = np.zeros((n, len(DR_LEVELS)), dtype=bool)
    hit_hi = np.zeros((n, len(DR_LEVELS)), dtype=bool)
    tot_viol = 0
    ratings = df["rating"].to_numpy(dtype=np.int64)
    # The scheduler's own curve evaluated at the interval that ACTUALLY happened, so this
    # arm can be calibration-checked on the very rows it schedules. See the same function
    # in fsrs_arm.py for why the workload ratio is not interpretable without it: comparing
    # at equal NOMINAL desired retention only measures efficiency if both models deliver
    # the retention they claim.
    # get_rwkv_data already carries the next review's elapsed time and outcome per card.
    label_t = df["label_elapsed_seconds"].to_numpy(dtype=np.float64)
    label_y = df["label_y"].to_numpy(dtype=np.float64)
    has_next = df["has_label"].to_numpy().astype(bool)
    pred = np.zeros(n, dtype=np.float64)
    t0 = time.perf_counter()

    with torch.inference_mode():
        for i, row in enumerate(df.to_dict("records")):
            feats = proc.get_tensor(row)
            cid, nid = row["card_id"], row["note_id"]
            did, pid = row["deck_id"], row["preset_id"]
            for d, k in ((proc.card_states, cid), (proc.note_states, nid),
                         (proc.deck_states, did), (proc.preset_states, pid)):
                d.setdefault(k, None)

            heads = rnn.button_heads(
                feats, proc.card_states[cid], proc.note_states[nid],
                proc.deck_states[did], proc.preset_states[pid], proc.global_state)
            # the label horizon rides along on the same grid call, so the calibration
            # prediction is the EXACT curve value there rather than a grid interpolation
            probe_t = torch.cat([t_grid, torch.tensor([max(1.0, label_t[i])],
                                                      dtype=torch.float32)])
            curves = rnn.button_curves(heads, probe_t)          # (4, T+1), rectified
            row_curve = curves[ratings[i] - 1].to(torch.float64).numpy()
            pred[i] = row_curve[-1]
            ivl[i], hit_lo[i], hit_hi[i], nv = invert(row_curve[:-1], log_t, DR_LEVELS)
            tot_viol += nv

            out = rnn.review(
                feats, proc.card_states[cid], proc.note_states[nid],
                proc.deck_states[did], proc.preset_states[pid], proc.global_state)
            (proc.card_states[cid], proc.note_states[nid], proc.deck_states[did],
             proc.preset_states[pid], proc.global_state) = out[5:10]

            if i and i % 2000 == 0:
                el = time.perf_counter() - t0
                print("  %d/%d  %.1f rev/s  eta %.1f min"
                      % (i, n, i / el, (n - i) / (i / el) / 60), flush=True)

    elapsed = time.perf_counter() - t0
    ivl_days = ivl / SECONDS_PER_DAY
    res = pd.DataFrame({
        "card_id": df["card_id"].to_numpy(),
        "review_th": df["review_th"].to_numpy(),
        "day_offset": df["day_offset"].to_numpy(),
        "rating": ratings,
        "pred": pred,
        "y": label_y,
        "has_next": has_next,
    })
    for k, dr in enumerate(DR_LEVELS):
        res["ivl_%d" % round(dr * 100)] = ivl_days[:, k]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_path, index=False)
    p = np.clip(pred[has_next], 1e-6, 1 - 1e-6)
    yy = label_y[has_next]
    logloss = float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())

    meta = {
        "user": user_id, "n_reviews": int(n), "n_cards": int(res["card_id"].nunique()),
        "ckpt": CHAMPION_CKPT, "env": APPLIED_ENV, "dr_levels": DR_LEVELS,
        "id_encoding_seed": ID_ENCODING_SEED, "n_grid": N_GRID, "s_max": float(rnn.s_max),
        "hit_lo_frac": [float(hit_lo[:, k].mean()) for k in range(len(DR_LEVELS))],
        "hit_hi_frac": [float(hit_hi[:, k].mean()) for k in range(len(DR_LEVELS))],
        "pava_monotonicity_violations": int(tot_viol),
        "reviews_per_second": float(n / elapsed),
        "n_scored": int(has_next.sum()),
        "logloss": logloss,
        "mean_pred": float(p.mean()),
        "mean_actual": float(yy.mean()),
        "calibration_bias": float(p.mean() - yy.mean()),
    }
    with open(str(out_path) + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("RWKV user %d: %d reviews in %.1f min (%.1f rev/s) -> %s"
          % (user_id, n, elapsed / 60, n / elapsed, out_path))
    print("  median ivl (days) by DR: " + "  ".join(
        "%d%%=%.3f" % (round(dr * 100), float(np.median(ivl_days[:, k])))
        for k, dr in enumerate(DR_LEVELS)))
    print("  unreachable-low frac:    " + "  ".join(
        "%d%%=%.4f" % (round(dr * 100), meta["hit_lo_frac"][k])
        for k, dr in enumerate(DR_LEVELS)))
    print("  PAVA non-monotone grid points: %d of %d" % (tot_viol, n * N_GRID))
    print("  ahead logloss %.4f on %d rows; mean pred %.4f vs actual %.4f (bias %+.4f)"
          % (logloss, int(has_next.sum()), p.mean(), yy.mean(), p.mean() - yy.mean()))


if __name__ == "__main__":
    main()
