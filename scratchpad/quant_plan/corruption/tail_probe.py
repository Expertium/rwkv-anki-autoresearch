"""Tail-energy probe: how much of a real card/note state's energy the rank-r truncation throws away.

WHY. The E1 lever (RWKV_QAT_TAIL_PENALTY, rwkv/model/rwkv_ops.py) adds

    lambda * mean over streams of  sum_{i > r} sigma_i^2 / sum_i sigma_i^2

to the training loss, where sigma are the singular values of the stream's own next state. That
quantity has NO natural scale: if the rank-r tail is 0.02 then lambda=1e-3 is invisible and
lambda=1e-2 is a 0.0002 loss term, while if the tail is 0.6 the same lambdas are 0.0006 and 0.006.
Picking lambda by feel would make the screen uninterpretable, so measure the tail on REAL states
first. That is all this probe does: it is measurement-only (monkey-patching, no source edits, no
change to any number the model computes), the same pattern as `finite_probe.py`.

WHAT IT RECORDS, per truncation rank (the stream's configured rank: 1 = the note in `l2i8s4`,
2 = the card): the number of states seen, the mean CUMULATIVE energy fraction at ranks 1..8
(cum_r = sum_{i<=r} sigma_i^2 / sum sigma_i^2, so the tail is 1 - cum_r), and the largest state
entry seen. The cumulative profile is the useful output: it says both what the truncation costs and
how fast the spectrum falls off, i.e. whether a penalty can realistically move it.

The state observed is the INPUT to `fake_lowrank_state` -- the pre-truncation state, which is exactly
what the penalty is computed on. It only exists on the PYTHON QAT path, so the driver sets
RWKV_QAT_FUSED=0 (the documented bit-parity twin of the fused CUDA kernels).

Installed by `tail_get_result.py`; driven by `measure_tail_energy.py`.
"""

import atexit
import json
import os
import sys

import torch

_OUT = os.environ.get("RWKV_TAIL_PROBE_OUT", "")
_NRANKS = 8
_ACC = {}          # "rank<r>" -> {"calls", "cum": [8 sums], "maxabs"}
_installed = False


def enabled() -> bool:
    return os.environ.get("RWKV_TAIL_PROBE", "0") == "1"


def record(s_BHKK, rank: int) -> None:
    with torch.no_grad():
        s = s_BHKK.reshape(-1, s_BHKK.shape[-2], s_BHKK.shape[-1]).float()
        sv = torch.linalg.svdvals(s)
        e = sv.square()
        tot = e.sum(dim=-1).clamp_min(1e-30)
        cum = torch.cumsum(e / tot.unsqueeze(-1), dim=-1)[:, :_NRANKS]  # (BH, 8)
        prof = cum.mean(dim=0)
        key = f"rank{rank}"
        a = _ACC.setdefault(key, {"calls": 0, "cum": [0.0] * _NRANKS, "maxabs": 0.0})
        a["calls"] += 1
        for i in range(_NRANKS):
            a["cum"][i] += float(prof[i])
        a["maxabs"] = max(a["maxabs"], float(s.abs().max()))


def summary() -> dict:
    out = {}
    for key, a in sorted(_ACC.items()):
        n = max(1, a["calls"])
        cum = [c / n for c in a["cum"]]
        r = int(key[4:])
        out[key] = {"calls": a["calls"], "cum_energy": cum, "maxabs": a["maxabs"],
                    "tail_at_own_rank": (1.0 - cum[r - 1]) if r - 1 < len(cum) else None}
    return out


def _dump() -> None:
    if not enabled():
        return
    doc = {"probe": "tail_energy", "n_states": sum(a["calls"] for a in _ACC.values()),
           "by_truncation_rank": summary()}
    path = _OUT or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tail_probe_out.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"[tail_probe] wrote {path} ({doc['n_states']} states)", flush=True)


def install() -> bool:
    global _installed
    if _installed:
        return True
    if not enabled():
        return False
    import rwkv.model.rwkv_ops as ops

    orig = ops.fake_lowrank_state

    def wrapper(s_BHKK, rank, factor_qmax, tag=""):
        if rank > 0:
            record(s_BHKK, rank)
        return orig(s_BHKK, rank, factor_qmax, tag)

    ops.fake_lowrank_state = wrapper
    atexit.register(_dump)
    _installed = True
    print(f"[tail_probe] installed (out={_OUT or 'tail_probe_out.json'})", flush=True)
    return True
