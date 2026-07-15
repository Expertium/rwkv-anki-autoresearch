"""Smoke test v2 for RWKV_PREHEAD_GATE (iter 16). The v1 smoke only called the ignored
method directly from Python and MISSED the real failure: a submodule call inside a
@torch.jit.ignore method crashes when invoked THROUGH scripted code ("'torch._C.ScriptModule'
object is not callable") -- the hollow-run bug. v2 therefore exercises head_and_out (a
SCRIPTED method) end-to-end, hook on and off, plus identity-at-init and gradient flow to the
gate Parameters. CPU-only."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
model.eval()  # dropout off so the identity comparison is exact
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
on = model.prehead_gate_on
print(f"constructed OK, jit={type(model).__mro__[1].__name__}, gate_on={on}, params={n}")

torch.manual_seed(0)
x = torch.randn(3, 5, 32)
# THE SCRIPTED PATH: head_and_out is a script method when JIT is on -- this is where the
# v1 bug fired ('ScriptModule' object is not callable inside the ignored body).
out_ahead, out_w, out_w_log_p, out_p = model.head_and_out(x)
print(f"scripted head_and_out OK: ahead {tuple(out_ahead.shape)}, p {tuple(out_p.shape)}")

if on:
    # identity at zero-init: gate output must equal input exactly
    y = model._apply_prehead_gate(x)
    assert torch.equal(x, y), "gate is not identity at zero-init"
    # gradient flow to the gate Parameters through the scripted path
    loss = out_ahead.sum() + out_p.sum() + out_w.sum()
    loss.backward()
    gw = model.prehead_gate_weight.grad
    gb = model.prehead_gate_bias.grad
    assert gw is not None and gb is not None, "no grad reached the gate parameters"
    print(f"identity-at-init PASS; grad flow PASS (|gw| {gw.abs().sum():.4f}, |gb| {gb.abs().sum():.4f})")
"""

def run(env_extra, label):
    env = dict(os.environ)
    env.update({"PYTHONPATH": REPO, "CUDA_VISIBLE_DEVICES": "",
                "RWKV_N_HEADS": "2", "RWKV_HEAD_DIM": "16"})
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, cwd=REPO,
                       capture_output=True, text=True)
    print(f"--- {label} (exit {r.returncode}) ---")
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2500:])
        sys.exit(1)

run({"RWKV_PREHEAD_GATE": "1", "RWKV_ZERO_FEATURES": "22"}, "gate ON + featmask, JIT on")
run({"RWKV_ZERO_FEATURES": "22"}, "gate OFF + featmask, JIT on")
run({"RWKV_PREHEAD_GATE": "1", "RWKV_ZERO_FEATURES": "22", "RWKV_NO_JIT": "1"}, "gate ON, NO_JIT")
print("ALL_PASS")
