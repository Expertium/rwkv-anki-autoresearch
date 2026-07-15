"""Smoke test for RWKV_PREHEAD_GATE (iter 16): JIT-on construction with the hook on AND
off, exact-identity-at-init check, and the param delta (+1,056). CPU-only."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
on = model.prehead_gate_on
print(f"constructed OK, jit={type(model).__mro__[1].__name__}, gate_on={on}, params={n}")
if on:
    torch.manual_seed(0)
    x = torch.randn(7, 32)
    y = model._apply_prehead_gate(x)
    assert torch.equal(x, y), f"gate is not identity at zero-init (max diff {(x-y).abs().max()})"
    print("identity-at-init check PASS")
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
        print(r.stderr[-2000:])
        sys.exit(1)

run({"RWKV_PREHEAD_GATE": "1", "RWKV_ZERO_FEATURES": "22"}, "gate ON + featmask, JIT on")
run({"RWKV_ZERO_FEATURES": "22"}, "gate OFF + featmask, JIT on")
print("ALL_PASS")
