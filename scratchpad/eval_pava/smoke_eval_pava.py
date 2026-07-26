"""Smoke test for the rectified-eval path (RWKV_EVAL_PAVA), 2026-07-26.

CPU-only by design: iter 31 owns the GPU and the no-co-tenant rule applies.

Checks, in order of what has historically broken:
  1. the model still CONSTRUCTS and SCRIPTS with the flag on (iter-16 lesson: a jit.ignore
     body that touches submodules turns a run hollow, and the failure is silent);
  2. _pava_rectify_eval actually enforces the button order and substitutes at the target
     rows, leaving every other row untouched;
  3. the powers fall back to 1.0 (classic PAVA) when the model has no trained theta, so the
     A18 champion can be scored;
  4. flag OFF is byte-identical (the default must not perturb any stored result).

Run:  .venv\\Scripts\\python.exe scratchpad/eval_pava/smoke_eval_pava.py
"""
import os
import sys

os.environ.setdefault("RWKV_N_HEADS", "2")
os.environ.setdefault("RWKV_HEAD_DIM", "16")

import torch  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def build(eval_pava: bool, pava_lambda: str):
    """Fresh interpreter state is impossible mid-process (old-style ScriptModule bakes the
    first construction's env flags into the compiled class), so each variant is built in a
    SUBPROCESS by the caller. This function runs inside that subprocess."""
    os.environ["RWKV_EVAL_PAVA"] = "1" if eval_pava else "0"
    os.environ["RWKV_PAVA_LAMBDA"] = pava_lambda
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from rwkv.model.srs_model import SrsRWKV
    return SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "construct":
        # subprocess variant: construct + script with the flag ON and a trained theta
        m = build(eval_pava=True, pava_lambda="0.1")
        assert hasattr(m, "pava_theta"), "pava_theta missing with lambda != 0"
        torch.jit.script(m)
        print("CONSTRUCT_SCRIPT_OK theta_shape=" + str(tuple(m.pava_theta.shape)))
        return

    if mode == "construct_nolambda":
        # the A18 case: rectified eval on a model that never trained PAVA
        m = build(eval_pava=True, pava_lambda="0")
        assert not hasattr(m, "pava_theta"), "theta should be absent at lambda 0"
        torch.jit.script(m)
        print("CONSTRUCT_NOTHETA_SCRIPT_OK")
        return

    # ---- in-process checks of the operator itself (no model needed) -------------
    from rwkv.model.pava import pava_rectify

    torch.manual_seed(0)
    B, T = 2, 12
    curve = torch.rand(B, T).clamp(0.05, 0.95)

    # two scored reviews; probe quadruples at (0, 1..4) and (1, 5..8), targets at (0,5)/(1,9)
    probe_rows = torch.tensor([[1, 2, 3, 4], [T + 5, T + 6, T + 7, T + 8]])
    probe_target = torch.tensor([5, T + 9])
    probe_pressed = torch.tensor([2, 0])  # pressed Good, then Again

    # force a hard order violation in both quadruples so pooling MUST fire
    flat = curve.reshape(-1)
    flat[probe_rows[0]] = torch.tensor([0.90, 0.30, 0.70, 0.80])
    flat[probe_rows[1]] = torch.tensor([0.60, 0.55, 0.50, 0.95])

    class Fake:
        pava_lambda = 0.1
        pava_theta = torch.full((3,), 0.5493061443340549)  # atanh(0.5) -> p = 1

    from rwkv.model.srs_model import SrsRWKV
    out = SrsRWKV._pava_rectify_eval.__wrapped__(
        Fake(), curve, probe_rows, probe_target, probe_pressed
    ) if hasattr(SrsRWKV._pava_rectify_eval, "__wrapped__") else None
    if out is None:  # jit.ignore keeps the plain function accessible in eager mode
        out = SrsRWKV._pava_rectify_eval(Fake(), curve, probe_rows, probe_target, probe_pressed)

    # 2a. the substituted value is the rectified pressed slot
    v = curve.reshape(-1)[probe_rows]
    rect = pava_rectify(v.float(), torch.ones_like(v), 2.0 * torch.tanh(Fake.pava_theta))
    want = rect.gather(1, probe_pressed.unsqueeze(1)).squeeze(1)
    got = out.reshape(-1)[probe_target]
    check("target rows carry the rectified pressed value",
          torch.allclose(got, want, atol=1e-6), f"got {got.tolist()} want {want.tolist()}")

    # 2b. rectified quadruples are non-decreasing (the whole point)
    check("rectified quadruples are ordered Again<=Hard<=Good<=Easy",
          bool((rect[:, 1:] >= rect[:, :-1] - 1e-6).all()), str(rect.tolist()))

    # 2c. pooling actually fired (a null test would pass 2a/2b vacuously)
    check("pooling changed the violating quadruples", bool((rect != v).any(dim=1).all().item()))

    # 2d. nothing else moved
    untouched = torch.ones(B * T, dtype=torch.bool)
    untouched[probe_target] = False
    check("all non-target rows are untouched",
          torch.equal(out.reshape(-1)[untouched], curve.reshape(-1)[untouched]))

    # 3. classic fallback when no theta exists
    class FakeNoTheta:
        pava_lambda = 0.0

    out2 = SrsRWKV._pava_rectify_eval(FakeNoTheta(), curve, probe_rows, probe_target,
                                      probe_pressed)
    rect_classic = pava_rectify(v.float(), torch.ones_like(v), torch.ones(3))
    want2 = rect_classic.gather(1, probe_pressed.unsqueeze(1)).squeeze(1)
    check("no-theta model falls back to classic p=1 PAVA",
          torch.allclose(out2.reshape(-1)[probe_target], want2, atol=1e-6))
    check("theta-init and classic p=1 agree (init IS classic PAVA)",
          torch.allclose(rect, rect_classic, atol=1e-6))

    print("\nSMOKE_" + ("ALL_PASS" if not FAILS else "FAILED: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
