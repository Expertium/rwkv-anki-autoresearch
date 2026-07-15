"""Smoke test for iter 18 (RWKV_ZERO_FEATURES=8,22 — duration + review-state ablation).
Same structure as iter 15's smoke: JIT-on construction, mask content, and a
features2card-level influence check for BOTH masked columns. CPU-only."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
mask_on = model.input_feat_mask_on
mask = model.input_feat_mask
zeros = [i for i in range(92) if mask[i] == 0]
print(f"constructed OK, jit={type(model).__mro__[1].__name__}, mask_on={mask_on}, zeros_at={zeros}")
assert zeros == [8, 22], zeros

torch.manual_seed(0)
for col in (8, 22):
    a = torch.randn(5, 92)
    b = a.clone(); b[:, col] += 100.0
    fa = model.features2card(model._apply_input_feat_mask(a))
    fb = model.features2card(model._apply_input_feat_mask(b))
    assert torch.equal(fa, fb), f"masked col {col} still influences features2card!"
    assert not torch.equal(model.features2card(a), model.features2card(b))
    print(f"col-{col} influence check PASS (masked: identical; unmasked: different)")
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

run({"RWKV_ZERO_FEATURES": "8,22"}, "8,22 ON, JIT on")
run({"RWKV_ZERO_FEATURES": "8,22", "RWKV_NO_JIT": "1"}, "8,22 ON, NO_JIT")
print("ALL_PASS")
