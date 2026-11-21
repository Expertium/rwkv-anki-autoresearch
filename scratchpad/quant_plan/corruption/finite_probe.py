"""Finiteness ("corruption") probe for the quantized eval path -- scratchpad-only, no source edits.

WHY THIS EXISTS (Andrew's gate, 2026-11-21): a candidate FAILS the corruption check if even ONE
card/note state value, or even ONE p(recall), is non-finite, on the 50 users with the longest
average review histories, with their histories NOT truncated. The eval's own guards cannot answer
that question:

  * `SrsRWKV._get_loss` returns None only on NaN of ONE probe tensor (get_result's NAN_SKIP path),
    and an +inf in a probability never trips it (softmax/sigmoid of an inf is finite);
  * the QAT state path already CAPS non-finite state entries before quantizing them
    (`_sanitize_state`, rwkv/model/rwkv_ops.py:313, added because a coarse int2 state used to run
    away to inf on a fragile card). That cap is exactly why a blow-up is invisible here: the raw
    state can reach inf, get capped to 1e4, and the run still looks healthy.

So the check must look at (a) the RAW recurrent state before the cap, (b) the token-shift vectors
before/after shift quantization, and (c) every returned p(recall) tensor.

HOW: monkey-patching, installed by `probe_get_result.py` (no repository source file is modified):

  * `rwkv_ops.fake_quant_state`      -> raw next-state input + quantized output (full-matrix int-N)
  * `rwkv_ops.fake_lowrank_state`    -> raw next-state input + rank-truncated output (the deploy
                                        rank-r path for card/note)
  * `rwkv_model.fake_quant_shift`    -> token-shift vector before/after shift quantization
  * `SrsRWKV.get_loss`               -> p_curve (ahead p(recall)), p_imm (immediate success p),
                                        p_imm_all (the 4-way immediate distribution) and the
                                        scalar losses, for EVERY row of every batch

The two state patches only fire on the PYTHON QAT path (`quant_aware_rwkv7`'s per-step loop), so the
driver sets RWKV_QAT_FUSED=0 -- the documented A/B parity switch (bit-parity with the fused CUDA
kernels) -- and RWKV_NO_JIT=1. A candidate whose state calls are 0 therefore did NOT have its state
observed: the driver treats that as an error, not as a pass.

Per user, one JSON line is appended to `$RWKV_FINITE_PROBE_DIR/probe_<pid>.jsonl`, rotated on
`get_result.get_benchmark_info` (called exactly once per user, before that user's batches are
scored), with a final `_summary` line at process exit.
"""

import atexit
import json
import os
import sys

import torch

_DIR = os.environ.get("RWKV_FINITE_PROBE_DIR", "")
_TAGS = ("state_raw", "state_quant", "lowrank_raw", "lowrank_quant", "shift_in", "shift_out",
         "p_curve", "p_imm", "p_imm_all", "loss_scalar", "stats_other")
_cur = None          # user being scored right now
_acc = None          # {"tag": {...}} accumulators for the current user
_total = {}          # process-wide accumulators (all users)
_path = None
_installed = False


def enabled() -> bool:
    return os.environ.get("RWKV_FINITE_PROBE", "0") == "1"


def _blank() -> dict:
    return {"calls": 0, "elems": 0, "nan": 0, "posinf": 0, "neginf": 0, "maxabs": 0.0,
            "first": None, "calls_by": {}}


def _acc_new() -> dict:
    return {t: _blank() for t in _TAGS}


def _out_path() -> str:
    d = _DIR or os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_out")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"probe_{os.getpid()}.jsonl")


def _emit(rec: dict) -> None:
    global _path
    if _path is None:
        _path = _out_path()
    with open(_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _scan(acc: dict, tag: str, t, extra: str = "") -> None:
    """Count non-finite entries of `t` into acc[tag]; keep the running |x|max and the first hit."""
    if t is None:
        return
    with torch.no_grad():
        a = t.detach()
        if not a.is_floating_point():
            return
        d = acc.setdefault(tag, _blank())
        d["calls"] += 1
        d["elems"] += a.numel()
        key = extra or "-"
        d["calls_by"][key] = d["calls_by"].get(key, 0) + 1
        finite = torch.isfinite(a)
        mx = float(torch.nan_to_num(a.abs(), nan=0.0).max()) if a.numel() else 0.0
        if mx > d["maxabs"]:
            d["maxabs"] = mx
        n_bad = int((~finite).sum())
        if n_bad:
            nan = int(torch.isnan(a).sum())
            inf = n_bad - nan
            d["nan"] += nan
            if inf:
                pos = int((a == float("inf")).sum())
                d["posinf"] += pos
                d["neginf"] += inf - pos
                kind = "posinf" if pos else "neginf"
            else:
                kind = "nan"
            if d["first"] is None:
                flat = int(torch.nan_to_num(a.abs(), nan=float("inf"),
                                            posinf=float("inf")).reshape(-1).argmax())
                d["first"] = {"user": _cur, "call": d["calls"], "kind": kind,
                              "flat_idx": flat, "value": float(a.reshape(-1)[flat]),
                              "extra": key}
        if d["first"] is None and mx == float("inf"):
            d["first"] = {"user": _cur, "call": d["calls"], "kind": "inf-in-maxabs",
                          "flat_idx": -1, "value": mx, "extra": key}


def _rotate(user_id) -> None:
    """Close the previous user's record and start this one."""
    global _cur, _acc
    _flush()
    _cur = int(user_id) if user_id is not None else None
    _acc = _acc_new()


def _flush() -> None:
    global _acc
    if _cur is None:
        return
    rec = {"user": _cur, "tags": {t: v for t, v in (_acc or {}).items() if v["calls"]}}
    _emit(rec)
    for t, v in (_acc or {}).items():
        if not v["calls"]:
            continue
        g = _total.setdefault(t, _blank())
        for k in ("calls", "elems", "nan", "posinf", "neginf"):
            g[k] += v[k]
        g["maxabs"] = max(g["maxabs"], v["maxabs"])
        if g["first"] is None and v["first"] is not None:
            g["first"] = v["first"]
        for kk, vv in v["calls_by"].items():
            g["calls_by"][kk] = g["calls_by"].get(kk, 0) + vv
    _acc = _acc_new()


def _summary() -> None:
    _flush()
    cfg = {k: os.environ[k] for k in sorted(os.environ) if k.startswith("RWKV_")}
    _emit({"_summary": True, "tags": _total, "env": cfg})


def verdict_of(paths) -> dict:
    """Aggregate probe_*.jsonl files into a verdict (used by check_corruption.py)."""
    per_user = {}
    total = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_summary"):
                for t, v in rec["tags"].items():
                    g = total.setdefault(t, _blank())
                    g["calls"] += v["calls"]
                    g["elems"] += v["elems"]
                    g["nan"] += v["nan"]
                    g["posinf"] += v["posinf"]
                    g["neginf"] += v["neginf"]
                    g["maxabs"] = max(g["maxabs"], v["maxabs"])
                continue
            u = rec["user"]
            pu = per_user.setdefault(u, {})
            for t, v in rec["tags"].items():
                d = pu.setdefault(t, _blank())
                d["calls"] += v["calls"]
                d["elems"] += v["elems"]
                d["nan"] += v["nan"]
                d["posinf"] += v["posinf"]
                d["neginf"] += v["neginf"]
                d["maxabs"] = max(d["maxabs"], v["maxabs"])
                if d["first"] is None:
                    d["first"] = v["first"]
    offenders = {u: {t: v for t, v in tags.items()
                     if v["nan"] or v["posinf"] or v["neginf"]}
                 for u, tags in per_user.items()}
    offenders = {u: v for u, v in offenders.items() if v}
    state_tags = ("state_raw", "state_quant", "lowrank_raw", "lowrank_quant", "shift_in", "shift_out")
    p_tags = ("p_curve", "p_imm", "p_imm_all")
    state_calls = sum(total.get(t, {}).get("calls", 0) for t in state_tags)
    state_bad = sum(total.get(t, {}).get("nan", 0) + total.get(t, {}).get("posinf", 0)
                    + total.get(t, {}).get("neginf", 0) for t in state_tags)
    p_bad = sum(total.get(t, {}).get("nan", 0) + total.get(t, {}).get("posinf", 0)
                + total.get(t, {}).get("neginf", 0) for t in p_tags)
    p_calls = sum(total.get(t, {}).get("calls", 0) for t in p_tags)
    return {"users": sorted(per_user), "per_user": per_user, "total": total,
            "offenders": offenders, "state_calls": state_calls, "state_bad": state_bad,
            "p_bad": p_bad, "p_calls": p_calls, "n_users": len(per_user),
            "pass": (state_bad == 0 and p_bad == 0)}


def install() -> bool:
    """Patch the state / shift / probability entry points. Idempotent."""
    global _installed, _acc
    if _installed:
        return True
    if not enabled():
        return False
    if _acc is None:
        _acc = _acc_new()

    import rwkv.model.rwkv_ops as ops
    import rwkv.model.rwkv_model as model_mod
    from rwkv.model.srs_model import SrsRWKV
    import rwkv.get_result as gr

    def wrap2(mod, name, tag_in, tag_out):
        orig = getattr(mod, name)

        def patched(t, *a, **kw):
            extra = kw.pop("_probe_extra", "")
            if not extra and a:
                extra = str(a[0])
            _scan(_acc, tag_in, t, extra)
            out = orig(t, *a, **kw)
            _scan(_acc, tag_out, out, extra)
            return out

        patched.__name__ = name
        patched.__doc__ = f"finite_probe wrapper around {name}"
        setattr(mod, name, patched)

    def wrap_passthrough(mod, name, tag_in, tag_out):
        """For fake_quant_shift(x, qmax): caller is the mixer, so the extra label is the qmax and
        the caller's layer_id (read off the frame -- diagnostics only)."""
        orig = getattr(mod, name)

        def patched(t, *a, **kw):
            extra = ""
            try:
                f = sys._getframe(1)
                self_ = f.f_locals.get("self")
                if self_ is not None:
                    extra = f"layer{getattr(self_, 'layer_id', '?')}"
            except Exception:
                pass
            if a:
                extra = (extra + ":" if extra else "") + f"qmax{a[0]}"
            _scan(_acc, tag_in, t, extra)
            out = orig(t, *a, **kw)
            _scan(_acc, tag_out, out, extra)
            return out

        patched.__name__ = name
        setattr(mod, name, patched)

    wrap2(ops, "fake_quant_state", "state_raw", "state_quant")
    wrap2(ops, "fake_lowrank_state", "lowrank_raw", "lowrank_quant")
    wrap_passthrough(model_mod, "fake_quant_shift", "shift_in", "shift_out")

    orig_get_loss = SrsRWKV.get_loss

    def get_loss_patched(self, batch, kd=None, kd_mix=None):
        stats = orig_get_loss(self, batch, kd=kd, kd_mix=kd_mix)
        if stats is not None:
            _scan(_acc, "p_curve", getattr(stats, "p_curve", None))
            _scan(_acc, "p_imm", getattr(stats, "p_imm", None))
            _scan(_acc, "p_imm_all", getattr(stats, "p_imm_all", None))
            _scan(_acc, "loss_scalar", getattr(stats, "average_loss", None))
            _scan(_acc, "loss_scalar", getattr(stats, "loss_tensor", None))
            for nm in ("ahead_avg", "ahead_raw_avg", "imm_avg", "w_loss_avg"):
                _scan(_acc, "stats_other", getattr(stats, nm, None))
        return stats

    SrsRWKV.get_loss = get_loss_patched

    orig_bench = gr.get_benchmark_info

    def bench_patched(db_path, db_size, user_id):
        if user_id != _cur:
            _rotate(user_id)
        return orig_bench(db_path, db_size, user_id)

    gr.get_benchmark_info = bench_patched

    atexit.register(_summary)
    _installed = True
    print(f"[finite_probe] installed (dir={_DIR or 'probe_out'}, pid={os.getpid()})", flush=True)
    return True
