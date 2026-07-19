"""A6 RWKV_STRIP_CMIX smoke (CPU): construction, params, scripted-forward branch.

Run WITH the A5 env + RWKV_STRIP_CMIX set (see the .cmd). Checks:
  A. full SrsRWKV constructs under JIT with the strip env; param count printed.
  B. off-path: a second process run WITHOUT the env must match A5's 2,115,359 exactly
     (checked by running this script twice -- the .cmd does both).
  C. micro scripted-forward branch test on one RWKV7Layer (CPU, T=8, reference kernel):
     stripped layer returns the time-mixer output unchanged; unstripped differs.
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
n = sum(p.numel() for p in model.parameters())
stripped = [(i, j) for i, m in enumerate(model.rwkv_modules)
            for j, b in enumerate(m.blocks) if b.cmix_stripped]
print(f"params: {n}")
print(f"stripped (module,layer): {stripped}")

if stripped:
    import dataclasses
    from rwkv.model.rwkv_model import RWKV7Layer
    # micro branch test: user stream config (module 4), layer 1
    cfg = DEFAULT_ANKI_RWKV_CONFIG.modules[4][1]
    torch.manual_seed(0)
    B, T, C = 2, 8, cfg.d_model
    x = torch.randn(B, T, C)
    v0 = torch.zeros(B, T, C)
    sel = torch.arange(T).unsqueeze(0).repeat(B, 1).clamp(min=0)
    skip = torch.zeros(B, T, dtype=torch.bool)

    lay_s = RWKV7Layer(cfg, 1)          # env on + user_id:1 in list -> stripped
    assert lay_s.cmix_stripped, "expected user_id:1 stripped"
    os.environ["RWKV_STRIP_CMIX"] = ""  # construct an unstripped twin
    lay_u = RWKV7Layer(cfg, 1)
    assert not lay_u.cmix_stripped
    lay_u.time_mixer.load_state_dict(lay_s.time_mixer.state_dict())  # share time-mixer weights
    with torch.no_grad():  # W_v is zero-init -> a fresh mixer contributes 0 (the documented
        lay_u.channel_mixer.W_v.weight.normal_(0, 0.1)  # zero-init smoke trap); randomize
    lay_s.eval(); lay_u.eval()
    with torch.no_grad():
        out_s, _ = lay_s(in_BTC=x, v0_BTC=v0, time_shift_select_BT=sel, skip_BT=skip)
        tm_out, _ = lay_s.time_mixer(in_BTC=x, v0_BTC=v0, time_shift_select_BT=sel, skip_BT=skip)
        out_u, _ = lay_u(in_BTC=x, v0_BTC=v0, time_shift_select_BT=sel, skip_BT=skip)
    same_as_tm = torch.equal(out_s, tm_out)
    differs = not torch.equal(out_u, out_s)
    print(f"stripped == time_mixer output: {same_as_tm}; unstripped differs: {differs}")
    assert same_as_tm and differs
    print("BRANCH_TEST_PASS")
print("SMOKE_DONE")
