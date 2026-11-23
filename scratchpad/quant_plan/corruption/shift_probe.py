"""Raw token-shift dump probe: does the PERSISTED shift vector really need 4 bits per coordinate?

WHY. The shift vectors are now the single largest term in a compressed state. With the fixed-basis
(codec) WKV coding at m=6 int8 the note spends 3*(6*6*8 + 64*4) = 1,632 bits, of which 3*256 = 768
are the shift vectors - 47% of the stream, and the same share of the card. Every shift rung so far
moved one bit-width at a time (int8 -> int6 -> int4 -> int3 -> int2) and the last two FAILED, so the
per-coordinate width is not a knob left. The open question is a different one: are the shift vectors
drawn from a SHARED low-dimensional subspace, the way the WKV states are? If they are, the shift is m
coefficients against a basis that ships with the model instead of 32 coordinates at int4 - the same
lever that beat the frontier for the WKV state, applied to the other half of the state.

WHAT IS PERSISTED. Per stream layer, TWO 32-d vectors: role 0 = the WKV mixer's token shift
(`t_xshift`, `RWKV7TimeMixer.forward`) and role 1 = the channel mixer's (`c_xshift`,
`RWKV7ChannelMixer.forward`). Both are `layer_norm(previous token's input)` gathered per timestep -
i.e. LayerNorm outputs, so zero-mean and unit-variance by construction before the learned gain.

This probe records the INPUT of `fake_quant_shift` (the pre-quantization persisted shift, which is
exactly what the deploy engine stores) for every call, sampled every RWKV_SHIFT_PROBE_EVERY-th row
up to RWKV_SHIFT_PROBE_MAX rows. Measurement-only: the original function is still called and still
returns its own value, so no number the model computes changes. Same pattern as `state_probe.py`.

It is installed by `shift_get_result.py` and driven by `measure_shift_subspace.py`. It patches the
module-level global, so it only works with the Python (RWKV_NO_JIT=1) path - a scripted forward would
have inlined the original.
"""

import atexit
import json
import os
import sys

import numpy as np
import torch

_DIR = os.environ.get("RWKV_SHIFT_PROBE_DIR", "")
_TAG = os.environ.get("RWKV_SHIFT_PROBE_TAG", "shift")
_STREAM = os.environ.get("RWKV_SHIFT_PROBE_STREAM", "x")
_EVERY = max(1, int(os.environ.get("RWKV_SHIFT_PROBE_EVERY", "1") or 1))
_MAX = int(os.environ.get("RWKV_SHIFT_PROBE_MAX", "4000") or 4000)

_ACC = {}      # key -> list of (C,) float32 vectors
_META = {}     # key -> {"stream", "role", "layer"}
_CALLS = {}    # key -> total rows seen (including skipped)
_N = {}        # key -> rows kept
_USERS = {}    # key -> user id per kept row
_SHAPES = {}   # key -> the (B,T,C) shapes seen
_CUR = [None]
_installed = False


def enabled() -> bool:
    return os.environ.get("RWKV_SHIFT_PROBE", "0") == "1"


def _caller(frame):
    """The (role, layer id) of the mixer whose forward called `fake_quant_shift`."""
    role, lay = "?", -1
    try:
        f = frame
        for _ in range(6):
            if f is None:
                break
            self_ = f.f_locals.get("self")
            if self_ is not None and hasattr(self_, "layer_id"):
                role = type(self_).__name__
                lay = int(getattr(self_, "layer_id"))
                break
            f = f.f_back
    except Exception:
        pass
    return role, lay


def record(x_BTC, role: str, lay: int) -> None:
    with torch.no_grad():
        if x_BTC.dim() != 3:
            return
        flat = x_BTC.reshape(-1, x_BTC.shape[-1]).detach().float().cpu().numpy()
        key = f"{_STREAM}|{role}|L{lay}"
        _SHAPES.setdefault(key, set()).add(tuple(x_BTC.shape))
        start = _CALLS.get(key, 0)
        _CALLS[key] = start + flat.shape[0]
        room = _MAX - _N.get(key, 0)
        if room <= 0:
            return
        # SPREAD the sample over the WHOLE call, not its first `room` rows: one call can carry far
        # more vectors than the cap (a (B,T,C) shift tensor is B*T rows), and a prefix would sample
        # only the first few items' histories, which is a real bias for a per-card statistic.
        n = flat.shape[0]
        stride = max(_EVERY, int(np.ceil(n / room)))
        idx = np.arange(0, n, stride)[:room]
        if idx.size == 0:
            return
        rows = flat[idx]
        _ACC.setdefault(key, []).extend(rows)
        _USERS.setdefault(key, []).extend([-1 if _CUR[0] is None else int(_CUR[0])] * len(idx))
        _META[key] = {"stream": _STREAM, "role": role, "layer": lay}
        _N[key] = _N.get(key, 0) + len(idx)


def out_path() -> str:
    d = _DIR or os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_probe_out")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"shifts_{_TAG}.npz")


def _dump() -> None:
    if not enabled() or not _ACC:
        return
    path = out_path()
    keys = sorted(_ACC)
    arrs = {f"a{i}": np.stack(_ACC[k]) for i, k in enumerate(keys)}
    arrs.update({f"u{i}": np.asarray(_USERS.get(k, []), dtype=np.int32) for i, k in enumerate(keys)})
    np.savez_compressed(path, **arrs)
    meta = {"probe": "shift_subspace", "tag": _TAG, "stream": _STREAM, "every": _EVERY,
            "max": _MAX, "keys": keys, "meta": {k: _META[k] for k in keys},
            "shapes": {k: sorted(_SHAPES[k]) for k in keys},
            "kept": {k: _N[k] for k in keys}, "rows": {k: _CALLS[k] for k in keys},
            "users": {k: sorted({int(v) for v in _USERS.get(k, [])}) for k in keys},
            "total_rows": int(sum(_N.values()))}
    with open(path[:-4] + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[shift_probe] wrote {path} ({meta['total_rows']} rows kept, {len(keys)} keys)",
          flush=True)


def install() -> bool:
    global _installed
    if _installed:
        return True
    if not enabled():
        return False
    import rwkv.model.rwkv_model as rm

    orig = rm.fake_quant_shift

    def wrapper(x_BTC, qmax):
        role, lay = _caller(sys._getframe(1))
        record(x_BTC, role, lay)
        return orig(x_BTC, qmax)

    wrapper.__name__ = "fake_quant_shift"
    rm.fake_quant_shift = wrapper

    # get_result calls this exactly once per user, before that user's batches are scored.
    import rwkv.get_result as gr
    orig_bench = gr.get_benchmark_info

    def bench_patched(db_path, db_size, user_id):
        _CUR[0] = None if user_id is None else int(user_id)
        return orig_bench(db_path, db_size, user_id)

    gr.get_benchmark_info = bench_patched
    atexit.register(_dump)
    _installed = True
    print(f"[shift_probe] installed (dir={_DIR or 'shift_probe_out'}, tag={_TAG}, "
          f"stream={_STREAM}, every={_EVERY}, max={_MAX})", flush=True)
    return True
