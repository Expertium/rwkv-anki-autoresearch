"""Smoke test for the CPU-inference 4-button rectified API (2026-07-26).

Third leg of Andrew's directive: "implement 'rectified, current-row duration zeroed on all
four' everywhere: training+eval+CPU inference". CPU-only (iter 31 owns the GPU).

The checks, ordered by what would actually bite in deployment:
  1. state_dict KEY SYMMETRY with the training model -- load_state_dict is strict, so a
     missing pava_theta means the deploy path cannot open a PAVA checkpoint at all;
  2. button_heads does not advance (or mutate) the incoming state -- the probes must be
     non-perturbative, exactly like the training skip rows;
  3. the four probe rows differ ONLY in the grade one-hot, with duration zeroed;
  4. the run() refactor onto curve_p is EXACT vs the old inline formula;
  5. ahead_residual's short-circuit is bit-identical to interp on zero logits;
  6. rectified curves are ordered across buttons at every t and decreasing in t;
  7. intervals are ordered and actually solve R(interval) = desired_retention.

Run:  .venv\\Scripts\\python.exe scratchpad/eval_pava/smoke_rnn_buttons.py
"""
import copy
import math
import os
import sys

# the merged-champion head recipe, minus the flags the RNN core does not implement
# (STRIP_CMIX / STRIP_L0_VLORA / STATE_CLAMP -- see the note printed at the end)
os.environ.setdefault("RWKV_N_HEADS", "2")
os.environ.setdefault("RWKV_HEAD_DIM", "16")
os.environ["RWKV_GRU_HEAD"] = "3"
os.environ["RWKV_PAVA_LAMBDA"] = "0.1"
os.environ["RWKV_NO_AHEAD_RESIDUAL"] = "1"
os.environ["RWKV_PROBE_DUR"] = "0.0"

import torch  # noqa: E402

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.model.srs_model_rnn import SrsRWKVRnn, _COL_DUR, _COL_R1  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    # detail only on FAIL: printing a failure explanation next to a PASS reads as if the
    # test were vacuous even when it was not
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   {detail}"))
    if not cond:
        FAILS.append(name)


def main():
    # deploy conditions: no autograd. (Also makes the state tensors leaves, which is what
    # RWKV7RNN.run's copy.deepcopy of the state requires -- inference is the only mode this
    # path is ever used in.)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    m = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG).float().eval()
    # A fixture with two jobs, in tension: the heads must be input-DEPENDENT or every button
    # gives the same curve and the ordering test is vacuous (the iter-20 lesson), but the
    # curves must also be REALISTIC or the interval solver just clamps at the bracket and
    # the solve test is vacuous too. So: random trunk, small random GRU weights, and GRU
    # BIASES at a sane prior -- which is also how the real head is initialised.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.startswith("gru_") or "pava" in n:
                continue
            p.normal_(0.0, 0.35 if p.dim() > 1 else 0.1)
        for n in ("gru_w_weight", "gru_s_weight", "gru_d_weight"):
            getattr(m, n).normal_(0.0, 0.02)
        m.gru_w_bias.zero_()
        m.gru_s_bias.copy_(torch.log(torch.tensor([1e4, 1e5, 1e6])))  # stabilities in s
        m.gru_d_bias.fill_(math.log(0.3))
        m.pava_theta.copy_(torch.tensor([0.9, -0.7, 0.4]))  # p = 2*tanh -> non-classic

    # ---- 1. state_dict key symmetry with the TRAINING model ---------------------
    from rwkv.model.srs_model import SrsRWKV
    tr = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    k_rnn, k_tr = set(m.state_dict()), set(tr.state_dict())
    check("pava_theta is in the RNN state_dict", "pava_theta" in k_rnn)
    check("RNN state_dict keys are a subset of the training model's",
          k_rnn <= k_tr, f"extra: {sorted(k_rnn - k_tr)[:5]}")
    missing = k_tr - k_rnn
    check("training model has no params the RNN path lacks",
          not missing, f"missing from RNN: {sorted(missing)[:8]}")

    # ---- build a state by running one real review -------------------------------
    feats = torch.randn(1, 92) * 0.3
    feats[:, _COL_R1:_COL_R1 + 4] = 0
    feats[:, _COL_R1 + 2] = 1  # pressed Good
    feats[:, _COL_DUR] = -0.35  # a REAL duration on the real row

    st = m.review(feats, None, None, None, None, None)
    states = st[5:]  # card, note, deck, preset, global

    # ---- 2. button_heads must not advance or mutate the state -------------------
    # the per-stream state is a nested dict {layer: ((x_shift, wkv), cmix_shift)}
    def same_state(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return set(a) == set(b) and all(same_state(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
        return a == b

    snap = copy.deepcopy(states)
    heads = m.button_heads(feats, *states)
    check("button_heads leaves the incoming state untouched (skip semantics)",
          same_state(states, snap))

    after = m.review(feats, *states)
    before = m.review(feats, *snap)
    check("a later review is unaffected by having asked for buttons",
          torch.equal(after[1], before[1]))

    # ---- 3. the probe rows differ only in the grade ------------------------------
    rows = []
    base = feats.clone()
    base[:, _COL_DUR] = 0.0
    base[:, _COL_R1:_COL_R1 + 4] = 0
    for k in range(4):
        r = base.clone()
        r[:, _COL_R1 + k] = 1
        rows.append(r)
    stacked = torch.cat(rows)
    others = [i for i in range(92) if not (_COL_R1 <= i < _COL_R1 + 4)]
    check("probe rows are identical outside the grade one-hot",
          bool((stacked[:, others] == stacked[0, others]).all()))
    check("probe duration is zeroed, not the real value",
          float(stacked[0, _COL_DUR]) == 0.0 and float(feats[0, _COL_DUR]) != 0.0)
    # and the API really uses them: heads must equal a manual per-row review
    manual = torch.cat([m.review(r, *states)[1] for r in rows])
    check("button_heads == manual per-button review from the same state",
          torch.equal(heads[1], manual))

    # ---- 4./5. the curve_p refactor is exact -------------------------------------
    t = torch.tensor([[3600.0], [86400.0], [864000.0]])
    a1, w1, s1, d1 = heads[0][:1], heads[1][:1], heads[2][:1], heads[3][:1]
    got = m.curve_p(a1, w1, s1, d1, t)
    raw = m.gru_forgetting_curve(w1, s1, d1, t)
    want = torch.sigmoid(
        torch.log(raw / (1 - raw)) + m.interp(a1.expand(t.shape[0], -1).contiguous(), t)
    )
    check("curve_p matches the old inline formula exactly", torch.equal(got, want))
    check("ahead_residual short-circuit is bit-identical to interp on zero logits",
          torch.equal(m.ahead_residual(a1, t),
                      m.interp(a1.expand(t.shape[0], -1).contiguous(), t)))

    # ---- 6. rectified curves: ordered across buttons, decreasing in t ------------
    grid = torch.tensor([60.0, 3600.0, 86400.0, 864000.0, 8640000.0])
    cur = m.button_curves(heads, grid)
    check("rectified curves ordered Again<=Hard<=Good<=Easy at every t",
          bool((cur[1:] >= cur[:-1] - 1e-6).all()), str(cur[:, 2].tolist()))
    check("each rectified curve is decreasing in t",
          bool((cur[:, 1:] <= cur[:, :-1] + 1e-6).all()))
    rawc = torch.stack([m.curve_p(heads[0][k:k + 1], heads[1][k:k + 1], heads[2][k:k + 1],
                                  heads[3][k:k + 1], grid.reshape(-1, 1)) for k in range(4)])
    check("rectification actually fired (raw curves were NOT already ordered)",
          bool((rawc[1:] < rawc[:-1] - 1e-6).any()),
          "raw already ordered -> ordering test is vacuous")

    # ---- 7. intervals ------------------------------------------------------------
    for r in (0.9, 0.8):
        iv = m.button_intervals(heads, desired_retention=r)  # asserts ordering internally
        at = m.button_curves(heads, iv).diagonal()
        det = f"iv={[round(x, 1) for x in iv.tolist()]} R={[round(x, 4) for x in at.tolist()]}"
        # no clamp escape hatch: with a sane prior the solver MUST land on the retention,
        # strictly inside the bracket, or it is not being exercised at all
        inside = bool(((iv > 1.01) & (iv < 0.99 * math.exp(m.s_max))).all())
        check(f"intervals for R={r} are strictly inside the bracket (solver ran)",
              inside, det)
        check(f"intervals solve R(t)={r}",
              bool(torch.allclose(at, torch.full((4,), r), atol=2e-3)), det)
        print(f"      R={r}: " + det)
    iv9 = m.button_intervals(heads, desired_retention=0.9)
    iv8 = m.button_intervals(heads, desired_retention=0.8)
    check("a lower retention target gives longer intervals", bool((iv8 > iv9).all()),
          f"{iv9.tolist()} vs {iv8.tolist()}")

    print("\nNOTE: rwkv_rnn_model.py implements none of RWKV_STRIP_CMIX / "
          "RWKV_STRIP_L0_VLORA / RWKV_STATE_CLAMP_* , so the PYTHON RNN path cannot run "
          "the merged (A18-trunk) champion yet -- same gaps 2/3/5 the Rust engine has.")
    print("SMOKE_" + ("ALL_PASS" if not FAILS else "FAILED: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
