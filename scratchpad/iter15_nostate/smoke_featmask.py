"""Smoke test for RWKV_ZERO_FEATURES (iter 15): JIT-on construction with the hook
on AND off (the iter-11 TorchScript lesson), mask content, and a features2card-level
check that a masked column stops influencing the output. CPU-only (CUDA_VISIBLE_DEVICES
is cleared by the caller) so it can run while A0 owns the GPU."""

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
print(f"constructed OK, jit={type(model).__mro__[1].__name__}, mask_on={mask_on}, "
      f"zeros_at={[i for i in range(92) if mask[i] == 0]}")

if mask_on:
    # column-22 influence check at the features2card level (the exact consumption point):
    # two inputs differing ONLY in col 22 must map to the same output once masked.
    torch.manual_seed(0)
    a = torch.randn(5, 92)
    b = a.clone(); b[:, 22] += 100.0
    fa = model.features2card(model._apply_input_feat_mask(a))
    fb = model.features2card(model._apply_input_feat_mask(b))
    assert torch.equal(fa, fb), "masked col 22 still influences features2card!"
    # and WITHOUT the mask they must differ (i.e. the check is not vacuous)
    assert not torch.equal(model.features2card(a), model.features2card(b))
    print("col-22 influence check PASS (masked: identical; unmasked: different)")
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

# hook ON + JIT on (default), hook OFF + JIT on, hook ON + NO_JIT
run({"RWKV_ZERO_FEATURES": "22"}, "hook ON, JIT on")
run({}, "hook OFF, JIT on")
run({"RWKV_ZERO_FEATURES": "22", "RWKV_NO_JIT": "1"}, "hook ON, NO_JIT")
print("ALL_PASS")
