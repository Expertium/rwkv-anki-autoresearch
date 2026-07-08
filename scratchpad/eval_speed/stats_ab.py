"""Old-vs-new exact-equality harness for the vectorized eval CPU path (extract_p, get_stats,
_eq_gather). Old implementations are loaded from the git-HEAD snapshots in this directory;
new ones from the live rwkv package. Every trial compares outputs for EXACT equality
(types, dtypes, values, json round-trips). Also times both on a 300k-review user.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


srs_old = load_module("srs_model_old", os.path.join(HERE, "srs_model_old_snippet.py"))
gr_old = load_module("get_result_old", os.path.join(HERE, "get_result_old.py"))
from rwkv.get_result import get_stats as get_stats_new, _eq_gather  # noqa: E402
from rwkv.model.srs_model import extract_p as extract_p_new  # noqa: E402

fails = 0


def make_stats(T, seed, all_label=False, no_query=False, all_query=False, with_dups=False):
    g = torch.Generator().manual_seed(seed)
    review_th = torch.arange(10, 10 + T, dtype=torch.long)
    if with_dups and T > 10:
        review_th[5] = review_th[4]  # duplicate key: later index must win in the dicts
    if all_label:
        has_label = torch.ones(T, dtype=torch.bool)
    else:
        has_label = torch.rand(T, generator=g) < 0.8
    if no_query:
        is_query = torch.zeros(T, dtype=torch.bool)
    elif all_query:
        is_query = torch.ones(T, dtype=torch.bool)
    else:
        is_query = torch.rand(T, generator=g) < 0.5
    return SimpleNamespace(
        label_review_th=review_th.unsqueeze(0),
        label_elapsed_seconds=(torch.rand(T, generator=g) * 1e6).float().unsqueeze(0),
        label_rating=torch.randint(0, 4, (T,), generator=g).unsqueeze(0),
        has_label=has_label.unsqueeze(0),
        is_query=is_query.unsqueeze(0),
        p_curve=torch.rand(T, generator=g).float().clamp(1e-4, 1 - 1e-4).unsqueeze(0),
        p_imm=torch.rand(T, generator=g).float().clamp(1e-4, 1 - 1e-4).unsqueeze(0),
        p_imm_all=torch.rand(T, 4, generator=g).float().unsqueeze(0),
        w=torch.rand(T, 8, generator=g).float().unsqueeze(0),
    )


def cmp_dicts(tag, d_old, d_new):
    global fails
    ok = True
    if list(d_old.keys()) != list(d_new.keys()):
        # key SETS must match; insertion order may differ only if old had reinsertions
        ok = set(d_old.keys()) == set(d_new.keys())
    for k in d_old:
        vo, vn = d_old[k], d_new[k]
        if type(vo) is not type(vn) or getattr(vo, "dtype", None) != getattr(vn, "dtype", None):
            ok = False
            break
        if not np.array_equal(np.asarray(vo), np.asarray(vn)):
            ok = False
            break
    if not ok:
        fails += 1
    print(f"  {tag:28s} n={len(d_old)}  {'PASS' if ok else 'FAIL'}")


def trial(name, T, seed, bins_as_int, **kw):
    global fails
    print(f"trial: {name} (T={T})")
    st = make_stats(T, seed, **kw)
    ds_old = srs_old.extract_p(st)
    ds_new = extract_p_new(st)
    cmp_dicts("extract_p.ahead_ps", ds_old.ahead_ps, ds_new.ahead_ps)
    cmp_dicts("extract_p.imm_ps", ds_old.imm_ps, ds_new.imm_ps)
    cmp_dicts("extract_p.imm_ps_all", ds_old.imm_ps_all, ds_new.imm_ps_all)
    cmp_dicts("extract_p.label_ratings", ds_old.label_ratings, ds_new.label_ratings)
    cmp_dicts("extract_p.elapsed", ds_old.label_elapsed_seconds, ds_new.label_elapsed_seconds)

    for mode, pred_dict in (("ahead", ds_new.ahead_ps), ("imm", ds_new.imm_ps)):
        eq = sorted(int(k) for k in pred_dict.keys())
        if len(eq) > 3:
            eq = eq[:: max(1, len(eq) // max(3, len(eq) * 3 // 4))] or eq
        if not eq:
            continue
        rng = np.random.default_rng(seed)
        raw_bins = rng.integers(0, 12, size=len(eq))
        rmse_bins_dict = {
            th: (int(b) if bins_as_int else float(b)) for th, b in zip(eq, raw_bins)
        }
        buf_o, buf_n = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_o):
            so, ro = gr_old.get_stats(7, eq, rmse_bins_dict, pred_dict, ds_new.label_ratings)
        with contextlib.redirect_stdout(buf_n):
            sn, rn = get_stats_new(7, eq, rmse_bins_dict, pred_dict, ds_new.label_ratings)
        ok = (
            json.dumps(so) == json.dumps(sn)
            and json.dumps(ro) == json.dumps(rn)
            and buf_o.getvalue() == buf_n.getvalue()
        )
        if not ok:
            fails += 1
            print(f"  OLD stats: {json.dumps(so)}")
            print(f"  NEW stats: {json.dumps(sn)}")
        print(f"  get_stats[{mode}] bins_int={bins_as_int}  n_eq={len(eq)}  {'PASS' if ok else 'FAIL'}")

        # run()-style raw gathers: _eq_gather vs the old per-th comprehension
        eq64 = np.asarray(eq, dtype=np.int64)
        old_el = [ds_new.label_elapsed_seconds[th].tolist() for th in eq]
        new_el = _eq_gather(ds_new.label_elapsed_seconds, eq64, "elapsed").tolist()
        if mode == "imm":
            old_pa = [ds_new.imm_ps_all[th].tolist() for th in eq]
            new_pa = _eq_gather(ds_new.imm_ps_all, eq64, "p_all").tolist()
        else:
            old_pa = new_pa = None
        ok2 = old_el == new_el and old_pa == new_pa
        if not ok2:
            fails += 1
        print(f"  raw gathers[{mode}]  {'PASS' if ok2 else 'FAIL'}")


trial("small mixed", 1000, 1, bins_as_int=True)
trial("small mixed float bins", 1000, 2, bins_as_int=False)
trial("all-label all-query", 5000, 3, bins_as_int=True, all_label=True, all_query=True)
trial("all-label no-query", 5000, 4, bins_as_int=True, all_label=True, no_query=True)
trial("dup review_th", 1000, 5, bins_as_int=True, with_dups=True)
trial("big mixed", 300000, 6, bins_as_int=True)

# timing on the big case
st = make_stats(300000, 7)
t0 = time.perf_counter()
ds_o = srs_old.extract_p(st)
t_old_ep = time.perf_counter() - t0
t0 = time.perf_counter()
ds_n = extract_p_new(st)
t_new_ep = time.perf_counter() - t0
eq = sorted(int(k) for k in ds_n.imm_ps.keys())
bins = {th: int(b) for th, b in zip(eq, np.random.default_rng(0).integers(0, 12, len(eq)))}
with contextlib.redirect_stdout(io.StringIO()):
    t0 = time.perf_counter()
    gr_old.get_stats(7, eq, bins, ds_n.imm_ps, ds_n.label_ratings)
    t_old_gs = time.perf_counter() - t0
    t0 = time.perf_counter()
    get_stats_new(7, eq, bins, ds_n.imm_ps, ds_n.label_ratings)
    t_new_gs = time.perf_counter() - t0
print(f"\ntiming 300k-review user: extract_p {t_old_ep*1e3:.0f} -> {t_new_ep*1e3:.0f} ms | "
      f"get_stats {t_old_gs*1e3:.0f} -> {t_new_gs*1e3:.0f} ms")
print("ALL_PASS" if fails == 0 else f"FAILURES={fails}")
sys.exit(0 if fails == 0 else 1)
