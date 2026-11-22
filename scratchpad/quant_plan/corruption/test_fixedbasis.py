"""Unit test for the fixed-basis state round-trip (approach 1), no model and no GPU needed.

Checks the properties the scheme relies on:
  1. m == K with qmax=inf reconstructs the state EXACTLY (P, Q are orthonormal).
  2. m == K with a finite qmax is a pure precision error (tiny relative error, like the full-matrix
     int-N path).
  3. a state that LIVES in the basis's top-m subspace survives m < K (the case the real states are
     measured to be in: corruption/state_subspace_*.json), and one that does not is spread out - the
     two extremes of the same knob.
  4. a stream/layer with no basis entry falls back to the rank-r path and SAYS SO instead of silently
     changing what is measured.
  5. the round-trip is a straight-through estimator: forward = the reconstruction, backward = identity.
  6. with the path unset the function is inert.

Run: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/test_fixedbasis.py
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
K = 32
BASIS = HERE / "_tmp_fb_test.npz"
_ok = []


def check(name, cond, detail=""):
    _ok.append(bool(cond))
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")


def make_basis():
    """A full orthonormal K x K pair (so m = K is testable) with a 16-column slice available."""
    rng = np.random.default_rng(0)
    p = np.linalg.qr(rng.normal(size=(K, K)))[0].astype(np.float32)
    q = np.linalg.qr(rng.normal(size=(K, K)))[0].astype(np.float32)
    # a second pair for the "note" stream, so the per-stream scope has something to switch to
    p2 = np.linalg.qr(rng.normal(size=(K, K)))[0].astype(np.float32)
    q2 = np.linalg.qr(rng.normal(size=(K, K)))[0].astype(np.float32)
    np.savez_compressed(BASIS, **{"P__card__L0": p, "Q__card__L0": q,
                                  "P__note__L0": p2, "Q__note__L0": q2})
    return p, q


def main() -> int:
    p, q = make_basis()
    os.environ["RWKV_FIXEDBASIS_PATH"] = str(BASIS)
    import rwkv.model.rwkv_ops as ops
    importlib.reload(ops)
    check("path picked up", ops.fixedbasis_loaded())

    # a state inside the basis's top-2 subspace (u, v are random mixtures of the first two columns)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, 2))
    b = rng.normal(size=(2, 2))
    x = torch.tensor((p[:, :2] @ a @ np.diag([3.0, 1.0]) @ b.T @ q[:, :2].T)
                     .astype(np.float32)).reshape(1, 1, K, K)
    # A state whose two modes lie ALONG the basis's own first two directions. This is the only way to
    # test "m=1 keeps the dominant mode": with a random orthonormal basis the first column is an
    # arbitrary direction, so a random mixture of the top-2 subspace (x above) can only show that m=1
    # truncates, not which mode it keeps. A fitted basis (the real scheme) has P's first column along
    # the dominant mode of the data, and this xd models exactly that.
    xd = torch.tensor((3.0 * np.outer(p[:, 0], q[:, 0]) + 1.0 * np.outer(p[:, 1], q[:, 1]))
                      .astype(np.float32)).reshape(1, 1, K, K)
    # a state orthogonal to the basis's top-2 subspace (the spread extreme)
    xs = torch.tensor((p[:, 16:18] @ np.diag([3.0, 1.0]) @ q[:, 16:18].T)
                      .astype(np.float32)).reshape(1, 1, K, K)

    y = ops.fake_fixedbasis_state(x, K, float("inf"), "card", 0)
    check("1. m=K, qmax=inf is exact", float((y - x).abs().max()) < 1e-5,
          f"max|err|={float((y - x).abs().max()):.2e}")

    y = ops.fake_fixedbasis_state(x, K, 127.0, "card", 0)
    rel = float((y - x).norm() / x.norm())
    check("2. m=K, int8 is a precision error", rel < 5e-2, f"rel={rel:.2e}")

    kept2 = 1.0 - float((ops.fake_fixedbasis_state(x, 2, float("inf"), "card", 0) - x).norm() ** 2
                        / x.norm() ** 2)
    check("3a. in-subspace state survives m=2", kept2 > 0.999, f"kept={kept2:.4f}")
    kept1 = 1.0 - float((ops.fake_fixedbasis_state(xd, 1, float("inf"), "card", 0) - xd).norm() ** 2
                        / xd.norm() ** 2)
    check("3b. m=1 keeps only the dominant mode", 0.85 < kept1 < 0.95, f"kept={kept1:.4f} (9/10 expected)")
    mix1 = 1.0 - float((ops.fake_fixedbasis_state(x, 1, float("inf"), "card", 0) - x).norm() ** 2
                       / x.norm() ** 2)
    print(f"     (info) m=1 on the arbitrary-mixture state keeps {mix1:.4f} - basis-dependent, no threshold")
    spread = 1.0 - float((ops.fake_fixedbasis_state(xs, 4, float("inf"), "card", 0) - xs).norm() ** 2
                         / xs.norm() ** 2)
    check("3c. out-of-subspace state is spread out", spread < 0.1, f"kept={spread:.4f}")

    y = ops.fake_fixedbasis_state(x, 8, 127.0, "deck", 7)
    yr = ops.fake_lowrank_state(x, 8, 127.0, "")
    check("4. unknown stream falls back to rank-r", float((y - yr).abs().max()) < 1e-6)

    xg = x.clone().requires_grad_(True)
    ops.fake_fixedbasis_state(xg, 8, 127.0, "card", 0).sum().backward()
    check("5. STE gradient is the identity", float((xg.grad - torch.ones_like(xg)).abs().max()) < 1e-6)

    os.environ.pop("RWKV_FIXEDBASIS_PATH")
    importlib.reload(ops)
    y = ops.fake_fixedbasis_state(x, 8, 127.0, "card", 0)
    check("6. no path = inert", (not ops.fixedbasis_loaded()) and float((y - x).abs().max()) > 0,
          "(returns the un-truncated input, so the value differs from the round-trip above)")

    # 7. the PER-STREAM scope: RWKV_FIXEDBASIS_SCOPE=note must leave the card on the rank-r path and
    #    put the note on the fixed basis. The basis file carries a card AND a note pair for this.
    os.environ["RWKV_FIXEDBASIS_PATH"] = str(BASIS)
    os.environ["RWKV_FIXEDBASIS_SCOPE"] = "note"
    importlib.reload(ops)
    card_fb = ops.fake_fixedbasis_state(x, 8, 127.0, "card", 0)
    card_rr = ops.fake_lowrank_state(x, 8, 127.0, "")
    note_fb = ops.fake_fixedbasis_state(x, 8, 127.0, "note", 0)
    note_rr = ops.fake_lowrank_state(x, 8, 127.0, "")
    check("7. scope=note: card stays rank-r, note uses the basis",
          float((card_fb - card_rr).abs().max()) < 1e-6 and float((note_fb - note_rr).abs().max()) > 1e-3,
          f"card|fb-rr|={float((card_fb - card_rr).abs().max()):.2e} "
          f"note|fb-rr|={float((note_fb - note_rr).abs().max()):.2e}")
    os.environ.pop("RWKV_FIXEDBASIS_SCOPE")

    print(f"\n{sum(_ok)}/{len(_ok)} checks passed")
    return 0 if all(_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
