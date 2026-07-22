"""A10 smoke (CPU): construction under the full A10 env, params, strip placement, branch test.

Run WITH the A10 env set (see run_track2_a10.cmd): RWKV_ARCH_MODULE=the user2 arch,
RWKV_STRIP_CMIX=10 entries, GRU head, L0-vlora strip, no-residual. Checks:
  A. full SrsRWKV constructs under JIT; param count printed + range-asserted
     (expected ~1,319,478 = A9 1,468,724 - user.L2 82,957 - 2 mixers->dummies 66,294).
  B. exactly 10 strips placed: card 1; deck 1/2/3; note 0; preset 0/1/2; user 0/1
     (module order card=0, deck=1, note=2, preset=3, user=4); user stream has 2 layers.
  C. micro branch test on the NEW note_id:0 strip (the note stream's only layer).
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
n = sum(p.numel() for p in model.parameters())
stripped = sorted((i, j) for i, m in enumerate(model.rwkv_modules)
                  for j, b in enumerate(m.blocks) if b.cmix_stripped)
n_user_layers = len(model.rwkv_modules[4].blocks)
print(f"params: {n}")
print(f"stripped (module,layer): {stripped}")
print(f"user layers: {n_user_layers}")

assert 1_316_000 < n < 1_323_000, f"param count {n} outside the expected A10 range"
expected = sorted([(0, 1), (1, 1), (1, 2), (1, 3), (2, 0), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1)])
assert stripped == expected, f"strip placement mismatch: {stripped} != {expected}"
assert n_user_layers == 2, "user stream must be 2 layers"

# C. branch test: note_id:0 (module 2 config, layer 0)
from rwkv.model.rwkv_model import RWKV7Layer
cfg = DEFAULT_ANKI_RWKV_CONFIG.modules[2][1]
torch.manual_seed(0)
B, T, C = 2, 8, cfg.d_model
x = torch.randn(B, T, C)
v0 = torch.zeros(B, T, C)
sel = torch.arange(T).unsqueeze(0).repeat(B, 1).clamp(min=0)
skip = torch.zeros(B, T, dtype=torch.bool)

lay_s = RWKV7Layer(cfg, 0)          # env on + note_id:0 in list -> stripped
assert lay_s.cmix_stripped, "expected note_id:0 stripped"
os.environ["RWKV_STRIP_CMIX"] = ""  # construct an unstripped twin
lay_u = RWKV7Layer(cfg, 0)
assert not lay_u.cmix_stripped
lay_u.time_mixer.load_state_dict(lay_s.time_mixer.state_dict())
with torch.no_grad():  # W_v is zero-init -> a fresh mixer contributes 0; randomize
    lay_u.channel_mixer.W_v.weight.normal_(0, 0.1)
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
print("SMOKE_ALL_PASS")
