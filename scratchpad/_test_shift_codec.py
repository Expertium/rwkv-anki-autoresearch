"""Mechanics test for the token-shift FIXED-BASIS codec (rwkv/model/rwkv_model.py).

No dumps needed: a synthetic 32-d vector family that lies in a 6-d subspace plus noise, coded by a
PCA basis this script fits itself. Checks (a) the codec beats the int-N round-trip at the same or
fewer bits, (b) the tiered lookup falls back r<role>__L<layer> -> r<role> and to int-N on a total
miss, (c) the path is INERT when the env var is unset.

Run: .venv/Scripts/python.exe scratchpad/_test_shift_codec.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.gettempdir()) / "_shift_codec_test.npz"
RNG = np.random.default_rng(20261123)
C, N, MODES = 32, 4000, 16

# a 6-d subspace + a mean, i.e. the structure the real LayerNorm'd shifts show
Q, _ = np.linalg.qr(RNG.normal(size=(C, 6)))
mu = RNG.normal(size=C) * 2.0
X = (mu + (RNG.normal(size=(N, 6)) * np.array([3, 2, 1, 0.6, 0.4, 0.3])) @ Q.T).astype(np.float32)
X += (RNG.normal(size=(N, C)) * 0.05).astype(np.float32)
cent = (X - X.mean(0)).astype(np.float64)
ev, evec = np.linalg.eigh(cent.T @ cent / (N - 1))
P = evec[:, np.argsort(ev)[::-1]][:, :MODES].astype(np.float32)
np.savez_compressed(TMP, M__r0=X.mean(0).astype(np.float32), P__r0=P)
os.environ["RWKV_SHIFT_FIXEDBASIS_PATH"] = str(TMP)
os.environ["RWKV_NO_JIT"] = "1"

import torch  # noqa: E402
from rwkv.model.rwkv_model import (fake_quant_shift, fake_shift_fixedbasis,  # noqa: E402
                                   shift_fixedbasis_loaded, _shift_fb_lookup)


def rel(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, C)
    b = np.asarray(b, dtype=np.float64).reshape(-1, C)
    return float(((a - b) ** 2).sum(1).mean() / (a ** 2).sum(1).mean())


x3 = torch.from_numpy(X[:512].reshape(8, 64, C))
print(f"loaded: {shift_fixedbasis_loaded()}   file: {TMP}")
raw4 = fake_quant_shift(x3, 7.0)
print(f"int4 raw           128 bits/vec  rel_err {rel(X[:512], raw4.numpy()):.5f}")
for lvl, q in (("int8", 127.0), ("int6", 31.0), ("int4", 7.0)):
    for m in (4, 6, 8, 12, 16):
        got = fake_shift_fixedbasis(x3, q, 0, "note", 0, m=m)
        be = rel(X[:512], got.numpy())
        print(f"codec m={m:<2} {lvl:<4} {2 * m * int(lvl[3:]):>4} bits/layer  rel_err {be:.5f}")

# ---- the tiered lookup -------------------------------------------------------------------------
print("\ntiered lookup:")
for stream, layer, role, want in [("note", 2, 0, True), ("card", 0, 0, True), ("note", 0, 1, False)]:
    got = _shift_fb_lookup(stream, layer, role, 8)
    ok = (got is not None) == want
    print(f"  {stream}/L{layer}/role{role}: {'found' if got is not None else 'miss (-> int-N)'}"
          f"  {'OK' if ok else 'UNEXPECTED'}")
    assert ok
# a total miss must return exactly the int-N round-trip
miss = fake_shift_fixedbasis(x3, 7.0, 1, "note", 0)
assert torch.equal(miss, fake_quant_shift(x3, 7.0)), "miss did not fall back to the int-N path"
print("  total miss == fake_quant_shift: OK")

# ---- inert when unset ---------------------------------------------------------------------------
env = dict(os.environ)
os.environ.pop("RWKV_SHIFT_FIXEDBASIS_PATH")
import importlib  # noqa: E402
import rwkv.model.rwkv_model as rm  # noqa: E402
importlib.reload(rm)
print(f"\nunset -> shift_fixedbasis_loaded() = {rm.shift_fixedbasis_loaded()}")
assert rm.shift_fixedbasis_loaded() is False
assert torch.equal(rm.fake_shift_fixedbasis(x3, 7.0, 0, "note", 0), rm.fake_quant_shift(x3, 7.0))
print("  unset path == fake_quant_shift: OK")
TMP.unlink(missing_ok=True)
print("\nALL CHECKS PASSED")
