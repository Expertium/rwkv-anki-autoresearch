"""Unit check for the E1 tail-energy penalty (RWKV_QAT_TAIL_PENALTY): formula, gradient, per-stream
attribution, drain, and the eval (empty-tag) path.

Why a separate check: the lever touches the QAT state path, which only runs under RWKV_NO_JIT on the
training device, so a broken formula would otherwise first show up as a slightly wrong loss in a
9-minute decay. Everything here is CPU-only and takes a second.

The penalty is defined as  tail = sum_{i > r} sigma_i^2 / sum_i sigma_i^2  per (B,H). A state built
with a KNOWN spectrum therefore has a known penalty: with sigma = [3, 2, 1] the rank-1 tail is
(4 + 1) / (9 + 4 + 1) = 5/14 and the rank-2 tail is 1/14.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/_test_tail_penalty.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ["RWKV_QAT_TAIL_PENALTY"] = "0.01"   # must be set BEFORE the module is imported
os.environ["RWKV_NO_JIT"] = "1"

import torch  # noqa: E402

import rwkv.model.rwkv_ops as ops  # noqa: E402


def synth_state(sigmas, seed=0):
    """A (1,1,K,K) state whose singular values are exactly `sigmas` (u, v random orthonormal)."""
    K = len(sigmas)
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(K, K, generator=g)
    b = torch.randn(K, K, generator=g)
    u, _ = torch.linalg.qr(a)
    v, _ = torch.linalg.qr(b)
    u = u.to(torch.float64)
    v = v.to(torch.float64)
    s = (u * torch.tensor(sigmas, dtype=torch.float64)) @ v.T
    return s.float().reshape(1, 1, K, K).requires_grad_(True)


fails = []


def check(name, got, want, tol=1e-6):
    ok = abs(float(got) - float(want)) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {float(got):.8f} want {float(want):.8f}")
    if not ok:
        fails.append(name)


def main():
    print(f"lever on: _TAIL_REG={ops._TAIL_REG}")

    # --- 1. the formula, from the state's own spectrum ------------------------
    s = synth_state([3.0, 2.0, 1.0])
    ops.accum_tail_penalty(s, 1, "card")
    got = ops.take_tail_penalty()
    check("rank-1 tail of sigma=[3,2,1]", got, 5.0 / 14.0)

    ops.accum_tail_penalty(s, 2, "card")
    check("rank-2 tail of sigma=[3,2,1]", ops.take_tail_penalty(), 1.0 / 14.0)

    # a rank-1 state has ZERO tail at rank 1 (the term is exactly the truncation loss)
    ops.accum_tail_penalty(synth_state([2.0, 0.0, 0.0]), 1, "card")
    check("rank-1 tail of a rank-1 state", ops.take_tail_penalty(), 0.0)

    # --- 2. the gradient reaches the state (the penalty is not inside no_grad) --
    s = synth_state([3.0, 2.0, 1.0])
    ops.accum_tail_penalty(s, 1, "card")
    t = ops.take_tail_penalty()
    t.backward()
    g = s.grad
    print(f"  {'ok  ' if g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 else 'FAIL'} "
          f"gradient reaches the state: grad_norm={0.0 if g is None else float(g.norm()):.6e}")
    if g is None or not torch.isfinite(g).all() or g.abs().sum() == 0:
        fails.append("gradient")

    # --- 3. per-stream attribution and the mean over streams -------------------
    ops.accum_tail_penalty(synth_state([3.0, 2.0, 1.0]), 1, "card")     # 5/14
    ops.accum_tail_penalty(synth_state([1.0, 1.0, 1.0]), 1, "note")     # 2/3
    check("mean over two streams", ops.take_tail_penalty(), (5.0 / 14.0 + 2.0 / 3.0) / 2.0)

    # --- 4. the drain clears, and returns None when nothing accumulated -------
    check("drain is empty after a drain", ops.take_tail_penalty() is None, 1.0, tol=1e-9) \
        if False else None
    empty = ops.take_tail_penalty()
    print(f"  {'ok  ' if empty is None else 'FAIL'} second drain returns None (accumulator cleared)")
    if empty is not None:
        fails.append("drain-clears")

    # --- 5. the eval path accumulates NOTHING (empty tag) ---------------------
    ops.fake_lowrank_state(synth_state([3.0, 2.0, 1.0]), 1, 127.0, "")
    print(f"  {'ok  ' if ops.take_tail_penalty() is None else 'FAIL'} "
          f"eval (empty tag) accumulates nothing")
    if ops.take_tail_penalty() is not None:
        fails.append("eval-tag")

    # --- 6. the training path DOES accumulate, and the STE forward is unchanged --
    st = synth_state([3.0, 2.0, 1.0])
    off = ops.fake_lowrank_state(st.detach().clone(), 1, 127.0, "")
    on = ops.fake_lowrank_state(st.detach().clone(), 1, 127.0, "card")
    same = torch.equal(off, on)
    print(f"  {'ok  ' if same else 'FAIL'} forward value identical with the lever on (pure STE)")
    if not same:
        fails.append("ste-forward")
    pen = ops.take_tail_penalty()
    print(f"  {'ok  ' if pen is not None else 'FAIL'} training (stream tag) accumulates")
    if pen is None:
        fails.append("train-tag")

    # --- 7. the QAT entry point: the tag reaches the penalty, output unchanged ---------------
    B, T, H, K = 1, 6, 1, 32
    g = torch.Generator().manual_seed(7)
    r = torch.randn(B, T, H, K, generator=g)
    k = torch.randn(B, T, H, K, generator=g)
    v = torch.randn(B, T, H, K, generator=g)
    w = -torch.rand(B, T, H, K, generator=g)
    a = torch.rand(B, T, H, K, generator=g)
    kd = torch.randn(B, T, H, K, generator=g)
    skip = torch.zeros(B, T, dtype=torch.bool)

    base = ops.quant_aware_rwkv7(r, k, v, w, a, kd, skip, float("inf"), 0, float("inf"))
    ops.take_tail_penalty()  # clear anything left over
    tagged = ops.quant_aware_rwkv7(r, k, v, w, a, kd, skip, float("inf"), 2, 127.0, "card")
    pen = ops.take_tail_penalty()
    print(f"  {'ok  ' if pen is not None and pen > 0 else 'FAIL'} "
          f"quant_aware_rwkv7 with a stream tag accumulates a positive penalty "
          f"({None if pen is None else float(pen):.6f})")
    if pen is None or pen <= 0:
        fails.append("qat-entry-accum")
    untagged = ops.quant_aware_rwkv7(r, k, v, w, a, kd, skip, float("inf"), 2, 127.0, "")
    same_out = torch.equal(tagged, untagged)
    print(f"  {'ok  ' if same_out else 'FAIL'} the tag does not change the returned output")
    if not same_out:
        fails.append("qat-entry-output")
    print(f"  {'ok  ' if ops.take_tail_penalty() is None else 'FAIL'} "
          f"an empty tag (eval) accumulates nothing through the entry point")
    if ops.take_tail_penalty() is not None:
        fails.append("qat-entry-eval")

    # --- 8. the lever being off is completely inert ---------------------------
    print("\nsecond pass with the lever OFF (fresh import):")
    code = (
        "import os, sys; sys.path.insert(0, r'%s'); os.environ.pop('RWKV_QAT_TAIL_PENALTY', None);"
        " os.environ['RWKV_NO_JIT']='1';"
        " import torch, rwkv.model.rwkv_ops as ops;"
        " print('  lever off:', ops._TAIL_REG);"
        " s = torch.randn(1,1,4,4);"
        " out = ops.fake_lowrank_state(s, 2, 127.0, 'card');"
        " print('  accumulated:', ops.take_tail_penalty());"
        " assert ops.take_tail_penalty() is None" % ROOT
    )
    import subprocess
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-400:])
    if r.returncode != 0:
        fails.append("lever-off-inert")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
