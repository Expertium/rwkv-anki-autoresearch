"""RWKV_CLOCK_AT_PREV_ANSWER=<T seconds> -- the fix for the query-row clock leak (2026-09-10).

DEFAULT OFF (unset, empty or "0"): the callers never enter this module, so every existing run is
byte-identical. Needs RWKV_ID_FEATURES=1 -- the leaking columns exist only in the -id layout.

THE LEAK (reported on Discord, confirmed on our own -id data: scratchpad/leak/PREREG.md). Every
clock column of the -id layout is measured to the reconstructed SHOW time `id - taken_millis`.
Anki CAPS taken_millis (60 s by default) and resets its timer when a card is re-shown, so the
reconstructed show time comes out LATE by the current review's excess -- and the excess predicts
the outcome (users 5001-5040: P(fail) 28.5% on capped reviews vs 18.0%; 2 s - 30 min gaps fail at
21-30% against 15.1% for the 71.4% of reviews shown < 1 s after the previous answer). A row that
predicts BEFORE the outcome must not see it. Two kinds of row do:
  * the QUERY row (is_query = 1): the imm prediction;
  * the PAVA PROBE rows: the four pre-answer button curves, i.e. the rectified ahead metric.
A REAL row is a finished review. Deploy rebuilds it from the same revlog fields, excess included,
so it is legitimate and is left untouched.
One LABEL carries a smaller copy: a real row's ahead target time is the interval to the card's
next review, measured to THAT review's reconstructed show time, so it carries the next review's
excess (1.64% of ahead targets move by >= 10%, almost all with intervals under 1 h).

THE FIX -- "the card appeared at the previous answer". For a query or probe row whose gap since the
previous answer on ANY card is below T (an in-session gap), delta = that gap; otherwise delta = 0
(a real break, which deploy sees too, is left alone). Every column measured to the show time moves
back by delta:
  * t_since_any_review -> 0;
  * the same-card interval, its cumulative, user tenure, deck age and sibling gap
    -> max(x - delta, 0), with the exact-0.0 "undefined" sentinels of the last two untouched;
  * the daily sin/cos pairs (interval, cumulative, time of day, time-of-day deviation) rotate back
    by delta.
A real row's label time t -> max(t - delta_k, 0), where delta_k is the in-session gap of the review
the label refers to (label_review_th). If that review is not in the chunk (it falls in the next
chunk), the label is left alone and COUNTED, so the fraction is visible rather than assumed.

DEPLOY applies the same rule to its live values: delta = now - (last answer on any card) if below
T, else 0; shift the query row's and the probe rows' clock columns by delta; evaluate the curve at
t - delta. At deploy the true in-session gap is ~0 anyway, so the rule mostly confirms what deploy
sees; it matters for a real short break (the user left the reviewer and came back), which it zeroes
exactly as training did. The Python deploy path (rwkv/run_as_rnn.py) calls shift_row_dict() and
ahead_t() below; the Rust engine consumes rows its caller built, so the Anki fork's feature builder
must apply the same rule (optimization/DEPLOY_FUNCTIONS.md).

THE UTC-DAY CALENDAR COLUMNS move only when the shift crosses UTC midnight, and on those rows they
DID leak (added 2026-09-10, before the leak-free WS): the moved row's time of day read 23:5x while
dow / doy / is_weekend / the review-time real cycles still read the next day, and that pair says the
excess was longer than the time since midnight. Users 5001-5040, T=1800: 259 such rows (0.013% of
query rows), P(fail) 0.29 against 0.14 on the other moved rows (scratchpad/leak/residual_channels.py).
All of them are functions of the UTC day index of the show time, and the time of day is UTC too, so
the crossing is detected exactly (up to bf16) from the STORED time-of-day pair: cross iff
delta > seconds since UTC midnight. A crossing row gets the previous day: each pair rotates back by
2*pi/period and is_weekend is re-derived from the rotated weekday. ⚠ doy on 1 January rotates to
angle 0 rather than to day 365/366 (the year length is not stored): 0.0043 rad off, below the bf16
resolution of the cosine, on 1/365 of crossing rows.

NOT SHIFTED, knowingly, because a stored row cannot recompute them -- and MEASURED, not assumed, on
the same users (residual_channels.py): the ANKI-day columns (elapsed_days and its cumulative,
day_offset_diff, cum_*_today; the rollover time is not stored) change on 0.0036% of query rows, and
the creation-batch counts (clipped at the show time) on 0.021%. Neither carries a measurable outcome
signal: P(fail) 0.10 (n=70) and 0.16 (n=399) against 0.14, i.e. 2.7e-7 and 3.3e-7 nats per query row,
which is the plug-in estimator's own bias floor. Closing them needs an LMDB rebuild.

Differences from phase L's counterfactual (scratchpad/leak/get_result_cf.py), which measured the
leak on an unchanged model: that one left tenure and deck age alone and never touched a label. Both
move here. Tenure and deck age are log-scale ages of weeks to years, so on nearly every row the
difference is below the bf16 resolution of the stored column.

The inverse of every scaled column is computed from its STORED value, which in the LMDB is bf16:
delta is recovered to ~1% (a 60 s gap +/- 0.6 s). A shift smaller than half a bf16 step rounds back
to the stored value, so long intervals are unchanged rather than perturbed.
"""
import os

import numpy as np
import torch

from rwkv import id_features as _idf

ENV = "RWKV_CLOCK_AT_PREV_ANSWER"

# global_labels column layout (data_processing.create_sample): les, led, label_y, label_rating,
# has_label, label_is_equalize, is_query
_LBL_T = 0
_LBL_HAS_LABEL = 4
_LBL_IS_QUERY = 6
_DAY = 86400.0


def threshold() -> float:
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "0"):
        return 0.0
    t = float(raw)
    if not t > 0.0:
        raise ValueError(f"{ENV}={raw!r}: expected a positive number of seconds, or unset")
    if not _idf.enabled():
        raise RuntimeError(
            f"{ENV} requires RWKV_ID_FEATURES=1: the leaking clock columns exist only in the -id "
            "layout, and on the published layout this flag would silently do nothing."
        )
    return t


T = threshold()


def enabled() -> bool:
    return T > 0.0


if enabled():
    from rwkv.data_processing import CARD_FEATURE_COLUMNS as _C, STATISTICS as _P

    _S = _idf.STATISTICS_ID
    _I_ANY = _C.index("scaled_t_since_any_review")
    _ANY = (_S["t_since_any_review_mean"], _S["t_since_any_review_std"])
    # (column index, (mean, std), has an exact-0.0 "undefined" sentinel)
    _SUB = [
        (_C.index("scaled_elapsed_seconds"),
         (_P["elapsed_seconds_mean"], _P["elapsed_seconds_std"]), False),
        (_C.index("scaled_elapsed_seconds_cumulative"),
         (_P["elapsed_seconds_cumulative_mean"], _P["elapsed_seconds_cumulative_std"]), False),
        (_C.index("scaled_user_tenure"), (_S["user_tenure_mean"], _S["user_tenure_std"]), False),
        (_C.index("scaled_deck_age_at_review"),
         (_S["deck_age_at_review_mean"], _S["deck_age_at_review_std"]), True),
        (_C.index("scaled_sibling_gap"), (_S["sibling_gap_mean"], _S["sibling_gap_std"]), True),
    ]
    _ROT = [(_C.index(s), _C.index(c)) for s, c in (
        ("elapsed_seconds_sin", "elapsed_seconds_cos"),
        ("elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos"),
        ("tod_sin", "tod_cos"),
        ("tod_dev_sin", "tod_dev_cos"))]
    # UTC-day calendar pairs, each with its period in days (see the docstring)
    _I_TOD = (_C.index("tod_sin"), _C.index("tod_cos"))
    _I_DOW = (_C.index("dow_sin"), _C.index("dow_cos"))
    _I_WKND = _C.index("is_weekend")
    _DAYROT = [(_I_DOW[0], _I_DOW[1], 7.0), (_C.index("doy_sin"), _C.index("doy_cos"), 365.25)]
    if _idf.real_cycles_enabled():
        for _p in _idf.CYCLE_PERIODS:
            if _p not in _idf._CYCLES_WITH_REAL_REVIEW_HALF:
                _t = _idf._cycle_tag(_p)
                _DAYROT.append((_C.index(f"cyc{_t}_sin"), _C.index(f"cyc{_t}_cos"), float(_p)))
    SHIFTED_COLUMNS = sorted({_I_ANY} | {i for i, _, _ in _SUB} | {i for p in _ROT for i in p}
                             | {i for a, b, _ in _DAYROT for i in (a, b)} | {_I_WKND})
    print(f"[clock-fix pid {os.getpid()}] ON: T={T:g} s -- query + probe rows moved to the previous "
          f"answer, label t minus the target's gap ({len(SHIFTED_COLUMNS)} columns)", flush=True)

_COUNT = {"calls": 0, "q_rows": 0, "q_shift": 0, "p_rows": 0, "p_shift": 0,
          "labels": 0, "l_shift": 0, "l_missing": 0, "x_day": 0}


def _inv(s, mean, std):
    """scaled -> seconds: the inverse of (log(1 + 1e-5 + x) - mean) / std."""
    return np.exp(s * std + mean) - 1.0 - 1e-5


def _fwd(x, mean, std):
    return (np.log(1.0 + 1e-5 + np.maximum(x, 0.0)) - mean) / std


def delta_from_scaled(s_any) -> np.ndarray:
    """Seconds to move a row back: its gap since the previous answer if it is an in-session gap
    (below T), else 0. The exact 0.0 of the stored column is the no-prior-review sentinel."""
    s = np.asarray(s_any, dtype=np.float64)
    g = np.maximum(_inv(s, *_ANY), 0.0)
    return np.where((s != 0.0) & (g < T), g, 0.0)


def shift_features(F: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """F (m, C) float64 scaled card features -> a copy with rows where delta > 0 moved back by
    delta. Rows with delta == 0 are returned unchanged."""
    out = F.copy()
    sh = delta > 0.0
    if not sh.any():
        return out
    d = delta[sh]
    out[sh, _I_ANY] = _fwd(0.0, *_ANY)
    for col, ms, sentinel in _SUB:
        x = F[sh, col]
        new = _fwd(_inv(x, *ms) - d, *ms)
        out[sh, col] = np.where(x == 0.0, x, new) if sentinel else new
    phi = -2.0 * np.pi * d / _DAY
    cp, sp = np.cos(phi), np.sin(phi)
    for s_col, c_col in _ROT:
        s0, c0 = F[sh, s_col], F[sh, c_col]
        out[sh, s_col] = s0 * cp + c0 * sp
        out[sh, c_col] = c0 * cp - s0 * sp
    # A shift past UTC midnight lands on the previous day: seconds since midnight come from the
    # STORED (unrotated) time-of-day pair.
    tod = np.mod(np.arctan2(F[sh, _I_TOD[0]], F[sh, _I_TOD[1]]), 2.0 * np.pi) * (_DAY / (2.0 * np.pi))
    cross = d > tod
    if cross.any():
        rows = np.nonzero(sh)[0][cross]
        for s_col, c_col, period in _DAYROT:
            ph = -2.0 * np.pi / period
            s0, c0 = F[rows, s_col], F[rows, c_col]
            out[rows, s_col] = s0 * np.cos(ph) + c0 * np.sin(ph)
            out[rows, c_col] = c0 * np.cos(ph) - s0 * np.sin(ph)
        k = np.mod(np.rint(np.arctan2(out[rows, _I_DOW[0]], out[rows, _I_DOW[1]]) * 7.0 / (2.0 * np.pi)), 7.0)
        out[rows, _I_WKND] = (k >= 5.0).astype(np.float64)   # Monday = 0, as in id_features
        _COUNT["x_day"] += int(cross.sum())
    return out


def _shift_tensor_rows(cf: torch.Tensor, rows: np.ndarray):
    """Shift the selected rows of a stored card-feature tensor. Unselected and unshifted rows are
    bit-identical; shifted rows are written back in the tensor's own dtype."""
    idx = np.nonzero(rows)[0]
    if idx.size == 0:
        return cf, 0
    a = cf[torch.from_numpy(idx)].double().numpy()
    delta = delta_from_scaled(a[:, _I_ANY])
    sh = delta > 0.0
    if not sh.any():
        return cf, 0
    b = shift_features(a, delta)
    out = cf.clone()
    out[torch.from_numpy(idx[sh])] = torch.from_numpy(b[sh]).to(cf.dtype)
    return out, int(sh.sum())


def _shift_labels(data):
    """Label time of each labelled REAL row minus the in-session gap of its target review."""
    lab = data.global_labels.double().numpy()
    sk = data.skips.numpy().astype(bool)
    real = ~sk
    lbl = real & (lab[:, _LBL_HAS_LABEL] > 0.5) & (lab[:, _LBL_IS_QUERY] < 0.5)
    n_lbl = int(lbl.sum())
    if n_lbl == 0:
        return data.global_labels, 0, 0, 0
    s_any = data.card_features[:, _I_ANY].double().numpy()
    rt = data.review_ths.numpy().astype(np.int64)
    real_rows = np.nonzero(real)[0]
    order = np.argsort(rt[real_rows], kind="stable")
    rt_s = rt[real_rows][order]
    d_s = delta_from_scaled(s_any[real_rows][order])
    rows = np.nonzero(lbl)[0]
    lrt = data.label_review_ths.numpy().astype(np.int64)[rows]
    pos = np.searchsorted(rt_s, lrt)
    pos_c = np.clip(pos, 0, max(rt_s.size - 1, 0))
    found = (pos < rt_s.size) & (rt_s[pos_c] == lrt)
    d = np.where(found, d_s[pos_c], 0.0)
    sh = d > 0.0
    n_missing = int((~found).sum())
    if not sh.any():
        return data.global_labels, n_lbl, 0, n_missing
    t_new = np.maximum(lab[rows[sh], _LBL_T] - d[sh], 0.0)
    out = data.global_labels.clone()
    out[torch.from_numpy(rows[sh]), _LBL_T] = torch.from_numpy(t_new).to(out.dtype)
    return out, n_lbl, int(sh.sum()), n_missing


def apply_to_sample(data):
    """Query rows moved to the previous answer + label times shifted. Call BEFORE insert_probes:
    probes copy the target's labels, so they inherit the shifted t, and the probe selection reads
    neither the features nor the label time."""
    import dataclasses
    lab = data.global_labels
    q = data.skips.numpy().astype(bool) & (lab[:, _LBL_IS_QUERY].float().numpy() > 0.5)
    cf, nq = _shift_tensor_rows(data.card_features, q)
    gl, nl, nls, nlm = _shift_labels(data)
    c = _COUNT
    c["calls"] += 1
    c["q_rows"] += int(q.sum()); c["q_shift"] += nq
    c["labels"] += nl; c["l_shift"] += nls; c["l_missing"] += nlm
    if c["calls"] % 500 == 1:
        print(f"[clock-fix pid {os.getpid()}] T={T:g}s chunks {c['calls']} query rows {c['q_rows']} "
              f"shifted {c['q_shift']} | labels {c['labels']} shifted {c['l_shift']} target not in "
              f"chunk {c['l_missing']} | probe rows {c['p_rows']} shifted {c['p_shift']} | day crossings "
              f"{c['x_day']}", flush=True)
    return dataclasses.replace(data, card_features=cf, global_labels=gl)


def apply_to_probes(data, meta):
    """Probe rows (meta.pos4) moved to the previous answer by their own gap -- a probe copies its
    target real row, whose features are left alone."""
    if meta is None:
        return data
    import dataclasses
    pm = np.zeros(data.card_features.size(0), dtype=bool)
    pm[np.asarray(meta.pos4).reshape(-1)] = True
    cf, n = _shift_tensor_rows(data.card_features, pm)
    _COUNT["p_rows"] += int(pm.sum()); _COUNT["p_shift"] += n
    return dataclasses.replace(data, card_features=cf)


# ---- the Python DEPLOY path (rwkv/run_as_rnn.py) ----

def shift_row_dict(row: dict) -> float:
    """In place: move a deploy QUERY row (a dict keyed by card-feature column) to the previous
    answer. Returns the delta applied, in seconds."""
    from rwkv.data_processing import CARD_FEATURE_COLUMNS as C
    F = np.array([[float(row[c]) for c in C]], dtype=np.float64)
    delta = delta_from_scaled(F[:, _I_ANY])
    if delta[0] <= 0.0:
        return 0.0
    G = shift_features(F, delta)
    for j in SHIFTED_COLUMNS:
        row[C[j]] = float(G[0, j])
    return float(delta[0])


def ahead_t(row):
    """The time at which the deploy path evaluates the stored curve for this review: the recorded
    interval, minus this review's in-session gap when the fix is on. Off: the value, unchanged."""
    t = row["elapsed_seconds"]
    if not enabled():
        return t
    d = float(delta_from_scaled(np.array([row["scaled_t_since_any_review"]]))[0])
    return max(float(t) - d, 0.0) if d > 0.0 else t
