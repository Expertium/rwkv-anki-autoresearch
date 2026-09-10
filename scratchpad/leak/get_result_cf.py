"""EVAL-ONLY counterfactual for the query-row clock leak (Discord, 2026-09-10). No production file is
edited: this wraps rwkv.get_result and patches rwkv.prepare_batch.insert_probes at import time, and
spawned fetch workers re-import this script first, so the patch reaches them too.

THE LEAK. Every clock column is computed from the reconstructed SHOW time `id - taken_millis`. Anki
caps taken_millis (60 s by default), and resets the timer when a card is re-shown, so on the QUERY row
(the imm prediction) and on the PAVA probe rows (the deploy button intervals, i.e. the rectified ahead
metric) the gap since the previous answer carries how the CURRENT review went. A deploy scheduler
predicts when the card appears, right after the previous answer, and sees ~0 s there.

THE COUNTERFACTUAL -- "the card appeared at the previous answer". On query rows and probe rows only,
delta = t_since_any_review in seconds if it is below LEAK_CF_T (an in-session gap), else 0 (a real
break, left alone). Every column measured to the same show time moves back by delta:
  t_since_any_review -> 0, sibling gap -> max(gap - delta, 0), same-card interval and its cumulative
  -> max(x - delta, 0), and the daily sin/cos pairs (interval, cumulative, tod, tod deviation) rotate
  back by delta. Real rows keep their values: they describe finished reviews, and deploy rebuilds them
  from the same revlog fields. Not covered: the ahead LABEL time (the next review's own excess).

Usage: set LEAK_CF_T=<seconds> (0 = no change: the harness control), then
       python scratchpad/leak/get_result_cf.py --config <eval.toml>
"""
import dataclasses
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "C:/Users/Andrew/rwkv-anki-autoresearch")
T_CF = float(os.environ["LEAK_CF_T"])           # required: no silent default

import rwkv.prepare_batch as _pbm               # noqa: E402
from rwkv.data_processing import CARD_FEATURE_COLUMNS as _C, STATISTICS as _P   # noqa: E402
from rwkv.id_features import STATISTICS_ID as _S  # noqa: E402

_I = {n: _C.index(n) for n in (
    "is_query", "scaled_t_since_any_review", "scaled_sibling_gap", "scaled_elapsed_seconds",
    "elapsed_seconds_sin", "elapsed_seconds_cos", "scaled_elapsed_seconds_cumulative",
    "elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos",
    "tod_sin", "tod_cos", "tod_dev_sin", "tod_dev_cos")}
_ANY = (_S["t_since_any_review_mean"], _S["t_since_any_review_std"])
_SIB = (_S["sibling_gap_mean"], _S["sibling_gap_std"])
_ES = (_P["elapsed_seconds_mean"], _P["elapsed_seconds_std"])
_ESC = (_P["elapsed_seconds_cumulative_mean"], _P["elapsed_seconds_cumulative_std"])
_STATS = {"calls": 0, "q_rows": 0, "q_shift": 0, "p_rows": 0, "p_shift": 0, "q_delta_sum": 0.0}


def _inv(s, ms):
    return np.exp(s * ms[1] + ms[0]) - 1.0 - 1e-5


def _fwd(x, ms):
    return (np.log(1.0 + 1e-5 + np.maximum(x, 0.0)) - ms[0]) / ms[1]


def transform(cf, rows):
    """Return (new cf, n shifted, sum of delta) with the counterfactual applied to `rows` (bool mask)."""
    idx = np.nonzero(rows)[0]
    if idx.size == 0 or T_CF <= 0:
        return cf, 0, 0.0
    a = cf[idx].double().numpy()                 # (m, F) float64 copy of the selected rows
    s_any = a[:, _I["scaled_t_since_any_review"]]
    g = np.maximum(_inv(s_any, _ANY), 0.0)
    delta = np.where((s_any != 0.0) & (g < T_CF), g, 0.0)    # exact 0.0 = the no-prior-review sentinel
    sh = delta > 0
    if not sh.any():
        return cf, 0, 0.0
    a[sh, _I["scaled_t_since_any_review"]] = _fwd(0.0, _ANY)
    sib = a[:, _I["scaled_sibling_gap"]]
    ok = sh & (sib != 0.0)                                   # exact 0.0 = undefined sibling gap
    a[ok, _I["scaled_sibling_gap"]] = _fwd(_inv(sib[ok], _SIB) - delta[ok], _SIB)
    for col, ms in (("scaled_elapsed_seconds", _ES), ("scaled_elapsed_seconds_cumulative", _ESC)):
        x = a[:, _I[col]]
        a[sh, _I[col]] = _fwd(_inv(x[sh], ms) - delta[sh], ms)
    phi = -2.0 * np.pi * delta / 86400.0                     # rotate each daily pair back by delta
    for s_col, c_col in (("elapsed_seconds_sin", "elapsed_seconds_cos"),
                         ("elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos"),
                         ("tod_sin", "tod_cos"), ("tod_dev_sin", "tod_dev_cos")):
        s0, c0 = a[:, _I[s_col]].copy(), a[:, _I[c_col]].copy()
        a[sh, _I[s_col]] = s0[sh] * np.cos(phi[sh]) + c0[sh] * np.sin(phi[sh])
        a[sh, _I[c_col]] = c0[sh] * np.cos(phi[sh]) - s0[sh] * np.sin(phi[sh])
    out = cf.clone()
    sel = torch.from_numpy(idx[sh])
    out[sel] = torch.from_numpy(a[sh]).to(cf.dtype)
    return out, int(sh.sum()), float(delta[sh].sum())


_orig_insert_probes = _pbm.insert_probes


def insert_probes_cf(data, density, base_seed, equalize_only=False):
    cf = data.card_features
    q = (cf[:, _I["is_query"]].float() > 0.5).numpy()
    cf2, nq, dq = transform(cf, q)
    data = dataclasses.replace(data, card_features=cf2)
    data2, meta = _orig_insert_probes(data, density, base_seed, equalize_only=equalize_only)
    np_rows, npsh = 0, 0
    if meta is not None:
        pm = np.zeros(data2.card_features.size(0), dtype=bool)
        pm[np.asarray(meta.pos4).reshape(-1)] = True
        np_rows = int(pm.sum())
        cf3, npsh, _ = transform(data2.card_features, pm)
        data2 = dataclasses.replace(data2, card_features=cf3)
    st = _STATS
    st["calls"] += 1
    st["q_rows"] += int(q.sum()); st["q_shift"] += nq; st["q_delta_sum"] += dq
    st["p_rows"] += np_rows; st["p_shift"] += npsh
    if st["calls"] % 100 == 1:
        print(f"[leak-cf pid {os.getpid()}] T={T_CF:g}s calls {st['calls']} query rows {st['q_rows']} "
              f"shifted {st['q_shift']} (mean delta {st['q_delta_sum'] / max(st['q_shift'], 1):.1f}s) "
              f"probe rows {st['p_rows']} shifted {st['p_shift']}", flush=True)
    return data2, meta


_pbm.insert_probes = insert_probes_cf
if os.environ.get("RWKV_EVAL_PAVA", "0") != "1":
    raise SystemExit("[leak-cf] RWKV_EVAL_PAVA must be 1: the patch runs inside insert_probes")

if __name__ == "__main__" and os.environ.get("LEAK_CF_SPAWN_TEST") == "1":
    # Proves the patch reaches a SPAWNED worker (Windows re-imports this script as __mp_main__).
    import multiprocessing
    sys.path.insert(0, "C:/Users/Andrew/rwkv-anki-autoresearch/scratchpad/leak")
    import spawn_probe
    _q = multiprocessing.Queue()
    _pr = multiprocessing.Process(target=spawn_probe.check, args=(_q,))
    _pr.start(); _name = _q.get(timeout=120); _pr.join()
    print(f"[leak-cf] spawned worker sees insert_probes = {_name}")
    raise SystemExit(0 if _name == "insert_probes_cf" else 49)
elif __name__ == "__main__":
    print(f"[leak-cf] counterfactual ON: T={T_CF:g} s on query + probe rows; production files untouched",
          flush=True)
    from rwkv.parse_toml import parse_toml
    import rwkv.get_result as _gr
    _gr.main(parse_toml())
