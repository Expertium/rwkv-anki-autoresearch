"""Smoke test for RWKV_MONO_CURVES (iter 22): the ahead-logit residual must come out
non-increasing along the time-point axis, equal exactly cummin(raw), keep the param count
unchanged, script under JIT, and pass gradients to ahead_linear.

One subprocess PER flag value: under the old-style ScriptModule API the FIRST construction's
flag value is captured into the compiled class method, so two models with different flags
cannot coexist in one process (bit us in this smoke's v1; production always has one flag
value per process). The raw (pre-projection) residual is recomputed manually through the
submodules (prehead_norm -> head_ahead_logits -> ahead_linear), bypassing head_and_out."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

on = os.environ.get("RWKV_MONO_CURVES", "0") == "1"
torch.manual_seed(0)
model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
model.eval()  # dropout inert so the manual recompute matches head_and_out exactly
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"constructed OK, base={type(model).__mro__[1].__name__}, mono={on}, params={n}")
assert n == 193724, f"param count changed: {n}"

# ahead_linear is ZERO-INIT in this arch (residual starts as exact zero -> the projection
# is a no-op at fresh init, same class of trap as iter 20's zero-init W_o). Randomize it
# so the projection is observable.
with torch.no_grad():
    model.ahead_linear.weight.normal_(std=0.5)
    model.ahead_linear.bias.normal_(std=0.5)

x = torch.randn(4, 7, 32)
ah, w, wlog, p = model.head_and_out(x)
with torch.no_grad():
    x_pre = model.prehead_norm(x)  # dropout is identity in eval
    raw = model.ahead_linear(model.head_ahead_logits(x_pre).float())

assert not (raw.diff(dim=-1) <= 0).all(), "raw residual already monotone -- vacuous test"
if on:
    assert (ah.diff(dim=-1) <= 1e-6).all(), "projected residual not non-increasing"
    assert torch.equal(ah, torch.cummin(raw, dim=-1)[0]), "head output != cummin(raw)"
    print("monotone PASS; ==cummin(raw) PASS; non-vacuous PASS")
    xg = torch.randn(4, 7, 32, requires_grad=True)
    ah_g, _, _, _ = model.head_and_out(xg)
    ah_g.sum().backward()
    g = model.ahead_linear.weight.grad
    assert g is not None and bool((g != 0).any()), "no grad reached ahead_linear"
    print("grad connectivity PASS")
    from rwkv.model.srs_model_rnn import SrsRWKVRnn
    rnn = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
    assert rnn.mono_curve_on, "RNN copy missing the flag"
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

run({"RWKV_MONO_CURVES": "1"}, "mono ON, JIT on")
run({}, "mono OFF, JIT on")
run({"RWKV_MONO_CURVES": "1", "RWKV_NO_JIT": "1"}, "mono ON, NO_JIT")
print("ALL_PASS")
