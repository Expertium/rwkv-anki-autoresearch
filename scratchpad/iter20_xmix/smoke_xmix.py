"""Smoke test for RWKV_XHEAD_MIX (iter 20), modeled on smoke_pregate.py v2 (iter-16 rules):
exercise the SCRIPTED RWKV7TimeMixer.forward end-to-end (hook on and off, JIT on and off),
identity-at-init, gradient flow to the mix Parameter, param count, and the
selective_cast + copy_downcast_ chain. CPU-only."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

mode = int(os.environ.get("RWKV_XHEAD_MIX", "0") or 0)
on = mode > 0
model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
model.eval()  # dropout off so identity comparisons are exact
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"constructed OK, base={type(model).__mro__[1].__name__}, mix_mode={mode}, params={n}")
# champion 193,724 + 14 layers * (v1: H*H*K=64 | v2: H*H*K*K=1024)
expect = {0: 193724, 1: 194620, 2: 208060}[mode]
assert n == expect, f"unexpected param count {n} (expected {expect})"

# THE SCRIPTED PATH: call a mixer THROUGH the (possibly scripted) module -- this is where
# the iter-16 class of bug fires ('ScriptModule' object is not callable in ignored bodies).
mixer = model.rwkv_modules[1].blocks[3].time_mixer  # deck L3 -- same shape everywhere
B, T, C = 2, 7, 32
torch.manual_seed(0)
x = torch.randn(B, T, C)
sel = torch.clamp(torch.arange(T) - 1, min=0).unsqueeze(0).expand(B, T).contiguous()
skip = torch.zeros(B, T, dtype=torch.bool)
out, v0 = mixer(in_BTC=x, v0_BTC=torch.zeros_like(x), time_shift_select_BT=sel, skip_BT=skip)
print(f"scripted time_mixer forward OK: out {tuple(out.shape)}")

if on:
    # identity at zero-init: the ignored helper must return its input exactly
    z = torch.randn(B, T, 2, 16)
    assert torch.equal(mixer._apply_xhead_mix(z), z), "mix not identity at zero-init"
    # W_o is ZERO-INIT in this arch -> at fresh init the mixer output is exactly in_BTC and
    # nothing upstream of W_o (incl. the mix) is observable or receives grad. Randomize W_o
    # to make the branch visible, then re-baseline.
    with torch.no_grad():
        mixer.W_o.weight.normal_(std=0.1)
    out, _ = mixer(in_BTC=x, v0_BTC=torch.zeros_like(x), time_shift_select_BT=sel, skip_BT=skip)
    # a nonzero delta must change the output (proves the branch is live in the scripted fwd)
    with torch.no_grad():
        mixer.xhead_mix_weight[0, 1] = 0.5  # v1: row of K; v2: KxK block
    out2, _ = mixer(in_BTC=x, v0_BTC=torch.zeros_like(x), time_shift_select_BT=sel, skip_BT=skip)
    assert not torch.equal(out, out2), "perturbed mix did not change scripted output"
    with torch.no_grad():
        mixer.xhead_mix_weight.zero_()
    # gradient flow through the scripted path
    xg = torch.randn(B, T, C, requires_grad=True)
    og, _ = mixer(in_BTC=xg, v0_BTC=torch.zeros_like(xg), time_shift_select_BT=sel, skip_BT=skip)
    og.sum().backward()
    g = mixer.xhead_mix_weight.grad
    assert g is not None and bool((g != 0).any()), "no/zero grad reached xhead_mix_weight"
    print("identity-at-init PASS; perturb-changes-output PASS; grad connectivity PASS")

# train_rwkv's model setup path (selective_cast walks modules; mix param lives on a submodule)
child = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG).selective_cast(torch.bfloat16)
child.copy_downcast_(model, dtype=torch.bfloat16)
if on:
    assert child.rwkv_modules[1].blocks[3].time_mixer.xhead_mix_weight.dtype == torch.bfloat16
print("selective_cast + copy_downcast_ OK")
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

run({"RWKV_XHEAD_MIX": "1", "RWKV_ZERO_FEATURES": "22"}, "mix v1 + featmask, JIT on")
run({"RWKV_ZERO_FEATURES": "22"}, "mix OFF + featmask, JIT on")
run({"RWKV_XHEAD_MIX": "1", "RWKV_ZERO_FEATURES": "22", "RWKV_NO_JIT": "1"}, "mix v1, NO_JIT")
run({"RWKV_XHEAD_MIX": "2", "RWKV_ZERO_FEATURES": "22"}, "mix v2 (KxK) + featmask, JIT on")
run({"RWKV_XHEAD_MIX": "2", "RWKV_ZERO_FEATURES": "22", "RWKV_NO_JIT": "1"}, "mix v2, NO_JIT")
print("ALL_PASS")
