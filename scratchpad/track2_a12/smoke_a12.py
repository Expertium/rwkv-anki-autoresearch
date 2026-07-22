"""A12 smoke (CPU): preset 3L->2L on the A9 champion base. STRIP_CMIX = A9's list minus
preset_id:2 (leaves with the layer) = 8 entries. Expected params 1,385,767 = A9
1,468,724 - preset.L2 (82,952 + dummy ~5). Run with the A12 env (see the .cmd)."""
import os
import sys

sys.path.insert(0, os.getcwd())

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
n = sum(p.numel() for p in model.parameters())
stripped = sorted((i, j) for i, m in enumerate(model.rwkv_modules)
                  for j, b in enumerate(m.blocks) if b.cmix_stripped)
print(f"params: {n}")
print(f"stripped (module,layer): {stripped}")

expected_strips = sorted([(0, 1), (1, 1), (1, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)])
assert stripped == expected_strips, f"strip placement mismatch: {stripped} != {expected_strips}"
assert 1_382_000 < n < 1_389_000, f"param count {n} outside the expected A12 range"
assert len(model.rwkv_modules[3].blocks) == 2, "preset stream must be 2 layers"
assert len(model.rwkv_modules[4].blocks) == 3, "user stream must stay 3 layers"
print("SMOKE_ALL_PASS")
