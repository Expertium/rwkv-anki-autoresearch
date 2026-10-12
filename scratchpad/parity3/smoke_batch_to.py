"""PreparedBatch.to(device) must carry EVERY declared field.

Written 2026-09-22, after `to()` silently dropped `selffeed_prev` / `selffeed_srcq` and cost a
9.5 h box arm that returned a bit-identical "null". The failure had no symptom: the fields carry
`None` defaults, so the hand-written constructor call compiled and the model's `selffeed is None`
branch is a legitimate code path (it IS the flag-off behaviour).

This check is deliberately GENERIC -- it enumerates the dataclass rather than naming fields -- so
it keeps working for the next field somebody adds, which is the whole point.

NON-VACUITY: the same checker is run against a deliberately broken `to()` that drops one field,
and must FAIL there. A field-coverage test that only ever passes proves nothing.

    .venv/Scripts/python.exe scratchpad/parity3/smoke_batch_to.py
"""
import dataclasses
import os
import sys

import torch

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
sys.path.insert(0, os.getcwd())
from rwkv.model.srs_model import PreparedBatch  # noqa: E402


def sample():
    """A batch with a DISTINCT non-None value in every field, so a drop cannot hide."""
    t = lambda v: torch.tensor([float(v)])
    i = lambda v: torch.tensor([int(v)], dtype=torch.long)
    kw = {}
    for n, f in enumerate(dataclasses.fields(PreparedBatch), start=1):
        a = str(f.type)
        if f.name == "num_data":
            kw[f.name] = 3
        elif "list[list[int]]" in a or "sub_gather_lens" == f.name:
            kw[f.name] = [[n]]
        elif "list[list[torch.Tensor]]" in a:
            kw[f.name] = [[i(n)]]
        elif "list[torch.Tensor]" in a:
            kw[f.name] = [i(n)]
        else:
            kw[f.name] = t(n)
    return PreparedBatch(**kw)


def check(move):
    """Return the list of fields `move` lost or corrupted."""
    b = sample()
    out = move(b)
    bad = []
    for f in dataclasses.fields(PreparedBatch):
        a, c = getattr(b, f.name), getattr(out, f.name, None)
        if c is None and a is not None:
            bad.append(f.name + " (dropped)")
        elif torch.is_tensor(a) and not (torch.is_tensor(c) and torch.equal(a, c)):
            bad.append(f.name + " (changed)")
    return bad


def broken(b):
    """The shape of the real bug: a hand-written rebuild that forgets the newest field."""
    names = [f.name for f in dataclasses.fields(PreparedBatch)]
    return PreparedBatch(**{n: getattr(b, n) for n in names[:-1]})


live = check(lambda b: b.to("cpu"))
ctrl = check(broken)
print("fields declared: %d" % len(dataclasses.fields(PreparedBatch)))
print("PreparedBatch.to('cpu') loses: %s" % (live or "nothing"))
print("the deliberately-broken rebuild loses: %s" % (ctrl or "nothing  <-- CHECK IS VACUOUS"))
ok = not live and len(ctrl) == 1
print("\n%s" % ("SMOKE PASS" if ok else "SMOKE FAIL"))
sys.exit(0 if ok else 1)
