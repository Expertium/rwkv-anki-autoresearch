"""A9 smoke (CPU): construction under the full A9 env, params, strip placement, branch test.

Run WITH the A9 env set (see run_track2_a9.cmd STEP -1): RWKV_ARCH_MODULE=the note1 arch,
RWKV_STRIP_CMIX=9 entries, GRU head, L0-vlora strip, no-residual. Checks:
  A. full SrsRWKV constructs under JIT; param count printed + range-asserted
     (expected ~1,468,724 = A8 1,617,975 - note.L1 82,957 - 2 mixers->dummies 66,294).
  B. exactly 9 strips placed: user 0/1/2, preset 0/1/2, deck 1/2, card 1 (module order
     card=0, deck=1, note=2, preset=3, user=4); note stream has 1 layer, mixer KEPT.
  C. micro branch test on the NEW user_id:0 strip (layer 0 = the v0-producing layer).
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
n_note_layers = len(model.rwkv_modules[2].blocks)
print(f"params: {n}")
print(f"stripped (module,layer): {stripped}")
print(f"note layers: {n_note_layers}")

assert 1_465_000 < n < 1_472_000, f"param count {n} outside the expected A9 range"
expected = sorted([(0, 1), (1, 1), (1, 2), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)])
assert stripped == expected, f"strip placement mismatch: {stripped} != {expected}"
assert n_note_layers == 1, "note stream must be 1 layer"

# C. branch test: user_id:0 (module 4 config, layer 0) stripped vs unstripped twin
from rwkv.model.rwkv_model import RWKV7Layer
cfg = DEFAULT_ANKI_RWKV_CONFIG.modules[4][1]
torch.manual_seed(0)
B, T, C = 2, 8, cfg.d_model
x = torch.randn(B, T, C)
v0 = torch.zeros(B, T, C)
sel = torch.arange(T).unsqueeze(0).repeat(B, 1).clamp(min=0)
skip = torch.zeros(B, T, dtype=torch.bool)

lay_s = RWKV7Layer(cfg, 0)          # env on + user_id:0 in list -> stripped
assert lay_s.cmix_stripped, "expected user_id:0 stripped"
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
