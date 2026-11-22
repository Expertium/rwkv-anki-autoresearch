"""Register the NOTE-int6 rungs stacked on the cheap shift rungs in rust_grid.CONFIGS.

`l2i6n` (card int8 + note int6, 3,200 bits) priced the note's int6 factors at +0.000268/+0.000307 as a
deploy screen. Nobody has stacked that on the note's int3/int2 SHIFT rungs, which are nearly free
(+0.000011/+0.000037 for int3, measured on row 35):

  l2i6s4n1i6s3  card rank-2 int6 (1,024) + note rank-1 int6 + int3 shifts (3x(384+192) = 1,728) = 2,752
  l2i6s4n1i6s2  card rank-2 int6 (1,024) + note rank-1 int6 + int2 shifts (3x(384+128) = 1,536) = 2,560

Both are below row 43's 2,880 and below chain 21's gated 2,944, so if the two costs compose as they did
for row 41 (sublinear) they are the smallest candidates that could still pass. Chain 17's `l2i6s4`
(card AND note at int6, 2,944) is what tells us whether the note's int6 survives the QAT regime.
"""
from pathlib import Path

P = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\rust_grid.py")
raw = P.read_bytes()

ANCHOR = (b'    "l2i5s4n1i8s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int5,note:1:int8",\r\n'
          b'                     "RWKV_QUANT_SHIFTS": "1",\r\n'
          b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},\r\n')

if b'"l2i6s4n1i6s3"' in raw:
    print("rust_grid.py: entries already present")
else:
    n = raw.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n} times, expected 1")
    block = (
        b'    # THE NOTE-int6 RUNGS UNDER THE CHEAP SHIFT RUNGS (2026-11-22). `l2i6n` priced the note\'s\r\n'
        b'    # int6 factors at +0.000268/+0.000307 on a deploy screen and row 35 priced the note\'s int3\r\n'
        b'    # shift rung at +0.000011/+0.000037, and row 41 showed those two-kind cuts compose SUBLINEARLY.\r\n'
        b'    # So the stack is the smallest plausible pass: card int6 (1,024) + note int6/int3 (1,728) = 2,752,\r\n'
        b'    # and with the note\'s int2 shifts (1,536) = 2,560. Both are below row 43\'s 2,880 (a FAIL).\r\n'
        b'    "l2i6s4n1i6s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int6",\r\n'
        b'                     "RWKV_QUANT_SHIFTS": "1",\r\n'
        b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},\r\n'
        b'    "l2i6s4n1i6s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int6",\r\n'
        b'                     "RWKV_QUANT_SHIFTS": "1",\r\n'
        b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},\r\n'
    )
    raw = raw.replace(ANCHOR, ANCHOR + block, 1)
    P.write_bytes(raw)
    print("rust_grid.py: 2 entries added")

import sys

sys.path.insert(0, str(P.parent))
import rust_grid as rg

for name in ("l2i5s4n1i8s2", "l2i6s4n1i6s3", "l2i6s4n1i6s2", "l2i5s4n1i8s3", "l2i4s4n1i8s3"):
    e = rg.CONFIGS[name]
    bad = [k for k in e if not k.startswith("RWKV_")]
    b = rg.bits_for(e)
    print(f"  {name:16s} keys={len(e)} bogus={bad} bits={b} total={sum(b.values())}")
print("CONFIGS size", len(rg.CONFIGS))