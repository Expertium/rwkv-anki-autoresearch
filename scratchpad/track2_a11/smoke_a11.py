"""A11 smoke (CPU): A10's arch (user 2L/card 2L/note 1L) with the note_id:0 strip
REMOVED from STRIP_CMIX (9 entries). Expected params 1,352,620 = A10 1,319,473 +
note.L0 mixer restored (+33,147). Run with the A11 env (see run_track2_a11.cmd)."""
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

expected_strips = sorted([(0, 1), (1, 1), (1, 2), (1, 3), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1)])
assert stripped == expected_strips, f"strip placement mismatch: {stripped} != {expected_strips}"
assert not model.rwkv_modules[2].blocks[0].cmix_stripped, "note.L0 mixer must be KEPT"
assert 1_349_000 < n < 1_356_000, f"param count {n} outside the expected A11 range"
assert len(model.rwkv_modules[4].blocks) == 2 and len(model.rwkv_modules[2].blocks) == 1
print("SMOKE_ALL_PASS")
