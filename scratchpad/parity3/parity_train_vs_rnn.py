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

out_train = train(x, tss, skip)                      # (1, T, C)

state = None
outs = []
for t in range(T):
    y, state = rnn.run(x[:, t], state)               # one timestep at a time
    outs.append(y)
out_rnn = torch.stack(outs, dim=1)

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
