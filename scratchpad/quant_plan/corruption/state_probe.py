"""Raw-state dump probe: save the ACTUAL card/note WKV states, not just their spectra.

WHY. Every state-size scheme in this lineage so far represents the 32x32 WKV state as a rank-r
outer product (U_r, V_r) and spends 2*r*K numbers per layer on it. That is only optimal if the two
factors have to be re-described from scratch at every step. They do not have to be: the state is a
decayed sum of the model's OWN key/value projections, so its dominant directions are plausibly
SHARED across users and time. If they are, a FIXED basis captures them for a fraction of the bits -
m coefficients instead of 2rK numbers - and the deploy compress step becomes two small matmuls
(faster than the current SVD/power iteration, so axis 3 gains too).

This probe answers that question from data. It is measurement-only: monkey-patching
`fake_lowrank_state`, no source edits, no change to any number the model computes, same pattern as
`finite_probe.py` and `tail_probe.py`.

WHAT IT SAVES. For every (stream, rank, layer) key, up to RWKV_STATE_PROBE_MAX raw 32x32 states
(every RWKV_STATE_PROBE_EVERY-th call), float32, to one .npz plus a .meta.json sidecar. That is
enough to estimate the state's covariance and therefore the captured-energy curve of any fixed-basis
scheme. The state observed is the INPUT to `fake_lowrank_state` (pre-truncation), which only exists
on the Python per-step loop, so the driver sets RWKV_QAT_FUSED=0.

Stream identity: `tail_tag` is EMPTY on the eval path (it is only filled while training with the
penalty on), so the key falls back to the rank and the layer id, which is read off the calling
frame. In `l2i8s4` rank 2 is the card (1 layer) and rank 1 is the note (3 layers).

Installed by `state_get_result.py`; driven by `measure_state_subspace.py`.
"""

import atexit
import json
import os
import sys

import numpy as np
import torch

_DIR = os.environ.get("RWKV_STATE_PROBE_DIR", "")
_TAG = os.environ.get("RWKV_STATE_PROBE_TAG", "state")
_EVERY = max(1, int(os.environ.get("RWKV_STATE_PROBE_EVERY", "1") or 1))
_MAX = int(os.environ.get("RWKV_STATE_PROBE_MAX", "4000") or 4000)

_ACC = {}      # key -> list of (32,32) float32 arrays
_META = {}     # key -> {"tag", "rank", "layer"}
_CALLS = {}    # key -> total calls seen (including skipped ones)
_N = {}        # key -> number of states kept
_USERS = {}    # key -> list of user ids, one per KEPT state (parallel to _ACC)
_CUR = [None]  # user being scored right now
_installed = False


def enabled() -> bool:
    return os.environ.get("RWKV_STATE_PROBE", "0") == "1"


def _layer_id() -> int:
    """Walk a few frames up looking for the mixer's `self.layer_id` (diagnostics only)."""
    try:
        f = sys._getframe(2)
        for _ in range(8):
            if f is None:
                break
            self_ = f.f_locals.get("self")
            if self_ is not None and hasattr(self_, "layer_id"):
                return int(getattr(self_, "layer_id"))
            f = f.f_back
    except Exception:
        pass
    return -1


def record(s_BHKK, rank: int, tag: str = "") -> None:
    with torch.no_grad():
        s = s_BHKK.reshape(-1, s_BHKK.shape[-2], s_BHKK.shape[-1])
        lay = _layer_id()
        key = f"{tag or 'x'}|r{rank}|L{lay}"
        _CALLS[key] = _CALLS.get(key, 0) + 1
        if (_CALLS[key] - 1) % _EVERY:
            return
        if _N.get(key, 0) >= _MAX:
            return
        w = s[0].detach().float().cpu().numpy().astype(np.float32)
    _ACC.setdefault(key, []).append(w)
    _USERS.setdefault(key, []).append(-1 if _CUR[0] is None else int(_CUR[0]))
    _META[key] = {"tag": tag, "rank": int(rank), "layer": lay}
    _N[key] = _N.get(key, 0) + 1


def out_path() -> str:
    d = _DIR or os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_probe_out")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"states_{_TAG}.npz")


def _dump() -> None:
    if not enabled() or not _ACC:
        return
    path = out_path()
    keys = sorted(_ACC)
    arrs = {f"a{i}": np.stack(_ACC[k]) for i, k in enumerate(keys)}
    arrs.update({f"u{i}": np.asarray(_USERS.get(k, []), dtype=np.int32) for i, k in enumerate(keys)})
    np.savez_compressed(path, **arrs)
    meta = {"probe": "state_subspace", "tag": _TAG, "every": _EVERY, "max": _MAX,
            "keys": keys, "meta": {k: _META[k] for k in keys},
            "kept": {k: _N[k] for k in keys}, "calls": {k: _CALLS[k] for k in keys},
            "users": {k: sorted({int(u) for u in _USERS.get(k, [])}) for k in keys},
            "total_states": int(sum(_N.values()))}
    with open(path[:-4] + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[state_probe] wrote {path} ({meta['total_states']} states, {len(keys)} keys)", flush=True)


def install() -> bool:
    global _installed
    if _installed:
        return True
    if not enabled():
        return False
    import rwkv.model.rwkv_ops as ops
    import rwkv.get_result as gr

    orig = ops.fake_lowrank_state

    def wrapper(s_BHKK, rank, factor_qmax, tag=""):
        if rank > 0:
            record(s_BHKK, rank, tag)
        return orig(s_BHKK, rank, factor_qmax, tag)

    wrapper.__name__ = "fake_lowrank_state"
    ops.fake_lowrank_state = wrapper

    # get_result calls this exactly once per user, before that user's batches are scored.
    orig_bench = gr.get_benchmark_info

    def bench_patched(db_path, db_size, user_id):
        _CUR[0] = None if user_id is None else int(user_id)
        return orig_bench(db_path, db_size, user_id)

    gr.get_benchmark_info = bench_patched
    atexit.register(_dump)
    _installed = True
    print(f"[state_probe] installed (dir={_DIR or 'state_probe_out'}, tag={_TAG}, "
          f"every={_EVERY}, max={_MAX})", flush=True)
    return True
