"""Smoke test for RWKV_NO_AHEAD_RESIDUAL (iter 22 redefined, Andrew 2026-07-16): with the
flag on, the ahead-logit residual out of head_and_out must be constant zeros (float32, right
shape), sit OUTSIDE autograd (no grad ever reaches the dead ahead head, live heads still
train), keep the param count unchanged, script under JIT, and leave the flag-off path
byte-identical.

One subprocess PER flag value: under the old-style ScriptModule API the FIRST construction's
flag value is captured into the compiled class method, so two models with different flags
cannot coexist in one process (production always has one flag value per process). The raw
(pre-disable) residual is recomputed manually through the submodules."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

on = os.environ.get("RWKV_NO_AHEAD_RESIDUAL", "0") == "1"
torch.manual_seed(0)
model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
model.eval()  # dropout inert so the manual recompute matches head_and_out exactly
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"constructed OK, base={type(model).__mro__[1].__name__}, nores={on}, params={n}")
assert n == 193724, f"param count changed: {n}"

# ahead_linear AND p_linear are ZERO-INIT in this arch -- randomize both so a LIVE head
# would emit nonzero and gradients can reach the trunk through p_linear (zero weights pass
# zero grad upstream; same trap class as iter 20's zero-init W_o).
with torch.no_grad():
    model.ahead_linear.weight.normal_(std=0.5)
    model.ahead_linear.bias.normal_(std=0.5)
    model.p_linear.weight.normal_(std=0.5)

x = torch.randn(4, 7, 32)
ah, w, wlog, p = model.head_and_out(x)
with torch.no_grad():
    x_pre = model.prehead_norm(x)  # dropout is identity in eval
    raw = model.ahead_linear(model.head_ahead_logits(x_pre).float())
assert not bool((raw == 0).all()), "raw residual all zero -- vacuous test"

if on:
    assert ah.shape == raw.shape and ah.dtype == torch.float32, f"shape/dtype: {ah.shape} {ah.dtype}"
    assert bool((ah == 0).all()), "residual not zeroed under RWKV_NO_AHEAD_RESIDUAL=1"
    print("zero-residual PASS; non-vacuous PASS")
    xg = torch.randn(4, 7, 32, requires_grad=True)
    ah_g, w_g, wlog_g, p_g = model.head_and_out(xg)
    assert not ah_g.requires_grad, "zero residual should be outside autograd"
    # p_linear (zero-init) still gets grad from raw-logit sum; w_linear's softmax outputs
    # have zero grad-of-sum at uniform, so probe liveness via p head + the trunk input.
    p_g.sum().backward()
    assert model.ahead_linear.weight.grad is None, "dead ahead head received grad"
    assert model.p_linear.weight.grad is not None and bool(
        (model.p_linear.weight.grad != 0).any()
    ), "no grad reached the live p head"
    assert xg.grad is not None and bool((xg.grad != 0).any()), "no grad reached the trunk"
    print("grad isolation PASS (p head + trunk live, ahead head dead)")
    from rwkv.model.srs_model_rnn import SrsRWKVRnn
    rnn = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
    assert rnn.no_ahead_residual, "RNN copy missing the flag"
    print("RNN-copy flag PASS")
else:
    assert torch.equal(ah, raw), "hook-off head output != raw (byte-identity broken)"
    print("hook-off byte-identity PASS")
"""

def run(env_extra, label):
    env = dict(os.environ)
    env.update({"PYTHONPATH": REPO, "CUDA_VISIBLE_DEVICES": "",
                "RWKV_N_HEADS": "2", "RWKV_HEAD_DIM": "16",
                "RWKV_ZERO_FEATURES": "22"})
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, cwd=REPO,
                       capture_output=True, text=True)
    print(f"--- {label} (exit {r.returncode}) ---")
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2500:])
        sys.exit(1)

run({"RWKV_NO_AHEAD_RESIDUAL": "1"}, "nores ON, JIT on")
run({}, "nores OFF, JIT on")
run({"RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_NO_JIT": "1"}, "nores ON, NO_JIT")
print("ALL_PASS")
