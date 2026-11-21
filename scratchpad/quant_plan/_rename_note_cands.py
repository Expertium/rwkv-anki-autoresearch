"""Fix the key collision: CANDIDATES already has an `n4` ("DIAGNOSTIC, over the bar: note int4", a
FULL-MATRIX note int4 entry defined later in the dict, so it silently won and my new rank-1 entry was
dead). Rename the new pair to `nr4` / `nr4s2` and assert there are no duplicate keys afterwards."""
from pathlib import Path

P = Path("scratchpad/quant_plan/quant_sweep.py")
with P.open(encoding="utf-8", newline="") as fh:
    text = fh.read()

pairs = [
    ('    "n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int4",',
     '    "nr4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int4",'),
    ('    "n4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int4",',
     '    "nr4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int4",'),
]
for old, new in pairs:
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new, 1)
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)

import re  # noqa: E402
names = re.findall(r'\n    "([^"]+)": \(', text)
dups = sorted({n for n in names if names.count(n) > 1})
print("entries:", len(names), "duplicates:", dups)
