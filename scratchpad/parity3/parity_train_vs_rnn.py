"""Numerical parity: the TRAINING stream (RWKV7, parallel form) vs the DEPLOY stream
(RWKV7RNN, recurrent form), on identical weights and inputs.

Written 2026-07-26 for Andrew's standing three-way-parity rule (CLAUDE.md sec 9). The rule
asks, of every change: what does training optimize, what does eval score, what will CPU
inference compute? Nothing was actually checking the third against the first at the
recurrence level, which is how RWKV_STRIP_CMIX / RWKV_STRIP_L0_VLORA reached the champion
recipe while existing only in rwkv_model.py -- the deploy twin silently computed a
DIFFERENT model, and no gate could catch it because each path was self-consistent alone.

The two forms are mathematically equivalent (RWKV-7's defining property), so on the same
weights they must agree to float noise. Each env combination runs in its OWN subprocess:
the old-style ScriptModule API bakes the first construction's env flags into the compiled
class, so two flag values cannot coexist in one process.

Run:  .venv\\Scripts\\python.exe scratchpad/parity3/parity_train_vs_rnn.py
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import dataclasses, os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.rwkv_model import RWKV7
from rwkv.model.rwkv_rnn_model import RWKV7RNN

STREAM = "card_id"
torch.manual_seed(0)
torch.set_grad_enabled(False)

base = DEFAULT_ANKI_RWKV_CONFIG.modules[0][1]
cfg = dataclasses.replace(base, n_layers=3, dropout=0.0, dropout_layer=0.0,
                          stream_name=STREAM)

train = RWKV7(config=cfg).float().eval()
rnn = RWKV7RNN(config=cfg).float().eval()

# 1. structural parity: the two must expose the SAME parameters, or the deploy path cannot
#    load a trained checkpoint (load_state_dict is strict)
kt, kr = set(train.state_dict()), set(rnn.state_dict())
assert kt == kr, f"state_dict mismatch\n  train-only: {sorted(kt-kr)}\n  rnn-only: {sorted(kr-kt)}"

# real weights: several params are zero-init by design (W_o, the scale linears), and with
# those left at zero large parts of the recurrence are identically zero -- parity would
# hold vacuously. Randomize everything, then copy train -> rnn.
for p in train.parameters():
    p.normal_(0.0, 0.25)
rnn.load_state_dict(train.state_dict())

strips = [b.cmix_stripped for b in train.blocks]
assert strips == [b.cmix_stripped for b in rnn.blocks], "strip maps differ"

C, T = cfg.d_model, 12
x = torch.randn(1, T, C) * 0.5
# contiguous stream, no skips: row t shifts from t-1, row 0 from itself (which is what the
# RNN does when its state is None)
tss = torch.tensor([[0] + list(range(T - 1))])
skip = torch.zeros(1, T, dtype=torch.bool)

# RWKV_RGATE (iter 55): the gate reads log elapsed SECONDS. Spread the timesteps across a
# realistic range -- ~1 s to ~9 years -- so rhat genuinely VARIES down the sequence instead of
# saturating to a constant, which is the difference between testing the gate and testing a bias.
# Key on the MODEL, not the env string: RWKV_RGATE names streams, so "note,deck" must leave this
# card_id stack ungated. Asserting the two agree is the scope-resolution test -- the analogue of
# the strip list's "other stream's strips are inert" case.
_scope = [s.strip().removesuffix("_id") for s in os.environ.get("RWKV_RGATE", "").split(",") if s.strip()]
rgate_on = any(nm.endswith("rgate_gain") for nm, _ in train.named_parameters())
assert rgate_on == (STREAM.removesuffix("_id") in _scope), (
    f"RWKV_RGATE={_scope} but this stream's gate present={rgate_on}")
log_dt = torch.linspace(0.0, 19.0, T).view(1, T) if rgate_on else None
if rgate_on:
    # `p.normal_()` above randomized rgate_log_s.bias too, which would park log_s near 0 and
    # push rhat to ~1e-4 for every ordinary gap -- a constant gate, i.e. a vacuous test. Put it
    # back in the responsive band on BOTH copies (named_parameters, so this does not depend on
    # indexing a ScriptModule's ModuleList).
    n_fixed = 0
    for m in (train, rnn):
        for nm, p in m.named_parameters():
            if nm.endswith("rgate_log_s.bias"):
                p.fill_(9.96)
                n_fixed += 1
    assert n_fixed == 2 * cfg.n_layers, f"expected one log_s bias per layer per copy, got {n_fixed}"

def run_train(dt):
    return train(x, tss, skip, dt) if rgate_on else train(x, tss, skip)

def run_rnn(dt):
    st = None
    o = []
    for t in range(T):
        y, st = rnn.run(x[:, t], st, dt[:, t] if dt is not None else None)
        o.append(y)
    return torch.stack(o, dim=1), st

out_train = run_train(log_dt)                        # (1, T, C)
out_rnn, state = run_rnn(log_dt)

d = (out_train - out_rnn).abs().max().item()
scale = out_train.abs().max().item()
print(f"stripped={strips} max|train-rnn|={d:.3e} (out scale {scale:.3f})")
assert scale > 1e-3, "outputs are ~zero -- parity would be vacuous"

tau = float(os.environ.get("RWKV_STATE_CLAMP_TAU", "0") or 0)
if tau > 0:
    # The training clamp is CUDA-only and window-aligned, so its numbers are not
    # reproducible here on purpose (see clamp_state's note). What IS checkable on CPU:
    #   huge tau -> inert, so parity must still hold exactly as if unclamped;
    #   small tau -> the carried state norm is actually bounded by tau.
    norms = [
        s[0][1].flatten(2).norm(p=2.0, dim=-1).max().item()
        for s in state.values()
    ]
    worst = max(norms)
    print(f"  tau={tau} worst carried ||S||={worst:.4f}")
    if tau < 1e3:
        assert worst <= tau * 1.0001, f"CLAMP FAIL: ||S||={worst} > tau={tau}"
        assert d > 1e-4, "clamp did not change anything -- test is vacuous at this tau"
        print("PARITY_OK")
        raise SystemExit(0)

assert d < 2e-5, f"PARITY FAIL: {d:.3e}"

if rgate_on:
    # NON-VACUITY, and it is the whole point of the case: agreement between two paths that both
    # IGNORE dt is not parity, it is a matched no-op. Feed a different elapsed-time vector and
    # require BOTH paths to move -- that is what proves dt actually reaches the recurrence
    # rather than being threaded to a parameter that happens to be zero.
    log_dt2 = torch.full_like(log_dt, 3.0)
    t2 = run_train(log_dt2)
    r2, _ = run_rnn(log_dt2)
    dt_move_train = (t2 - out_train).abs().max().item()
    dt_move_rnn = (r2 - out_rnn).abs().max().item()
    d2 = (t2 - r2).abs().max().item()
    print(f"  dt-sensitivity: train {dt_move_train:.3e}, rnn {dt_move_rnn:.3e}; "
          f"parity at dt2 {d2:.3e}")
    assert dt_move_train > 1e-4, "elapsed time does not reach the TRAINING recurrence"
    assert dt_move_rnn > 1e-4, "elapsed time does not reach the DEPLOY recurrence"
    assert d2 < 2e-5, f"PARITY FAIL at the second dt: {d2:.3e}"
print("PARITY_OK")
"""


def run(tag, env_extra):
    env = dict(os.environ, PYTHONPATH=REPO, **env_extra)
    print(f"--- {tag}: {env_extra or '(no flags)'}")
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env,
                       capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    for ln in out[-6:]:
        print("    " + ln)
    return p.returncode == 0 and any("PARITY_OK" in ln for ln in out)


def main():
    cases = [
        ("baseline", {}),
        ("strip L0 v_lora", {"RWKV_STRIP_L0_VLORA": "1"}),
        ("strip cmix L1", {"RWKV_STRIP_CMIX": "card_id:1"}),
        ("both, 2 mixers", {"RWKV_STRIP_L0_VLORA": "1",
                            "RWKV_STRIP_CMIX": "card_id:1,card_id:2"}),
        # a strip list naming a DIFFERENT stream must leave this one untouched -- catches a
        # match that ignores the stream name (which is what an unstamped config would do)
        ("other stream's strips are inert", {"RWKV_STRIP_CMIX": "user_id:1,deck_id:2"}),
        # the clamp: inert at a huge tau (parity must survive it), binding at a small one
        ("state clamp, huge tau (inert)", {"RWKV_STATE_CLAMP_TAU": "1e9"}),
        ("state clamp, small tau (binds)", {"RWKV_STATE_CLAMP_TAU": "0.05"}),
        # iter 54. The RNN file carried its OWN hardcoded square(relu(k)), so this flag is
        # precisely the shape §9 was written for: without a case here, train/eval would use
        # relu(k)^p and deploy relu(k)^2, each self-consistent in isolation.
        ("learnable cmix exponent", {"RWKV_CMIX_POW": "1"}),
        # iter 55. The gate is the first thing to feed a RAW INPUT FEATURE into the recurrence,
        # so both a plumbing bug and a formula divergence are possible; the case asserts parity
        # AND that changing dt moves both paths (see the non-vacuity block in CHILD).
        ("retrievability gate", {"RWKV_RGATE": "card"}),
        # a scope naming a DIFFERENT stream must leave this one ungated -- the same inertness
        # check the strip list gets, and the one that catches a scope match ignoring the name
        ("other stream's rgate is inert", {"RWKV_RGATE": "note,deck"}),
        # RWKV_INTERLEAVE (iter 41) has NO case here BY DESIGN: it lives at the SrsRWKV
        # level (multi-stream schedule + gather composition), which this single-stream
        # harness cannot see. Its coverage: scratchpad/parity3/smoke_interleave.py (the
        # depth-1 oracle proves the composition bit-exact vs the sequential branch) plus
        # the standard checkpoint-level trace parity (export_rnn_trace + verify) before
        # any interleaved champion ships -- the RNN mirror reads the same flag.
        #
        # RWKV_ID_FEATURES (the -id features rebuild) likewise has NO case here, and for the
        # same structural reason: it changes the INPUT WIDTH, which lives in SrsRWKV /
        # SrsRWKVRnn, one level above this single-stack harness. Its equivalent guard is
        # scratchpad/parity3/smoke_id_features_width.py, which asserts the training class, the
        # deploy RNN class and data_processing.CARD_FEATURE_COLUMNS all agree on the width
        # under BOTH flag values (92 / 112) -- the same three-way question in the same shape.
    ]
    ok = [run(t, e) for t, e in cases]
    print("\nPARITY_" + ("ALL_PASS" if all(ok) else "FAILED: "
                         + ", ".join(t for (t, _), o in zip(cases, ok) if not o)))
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
