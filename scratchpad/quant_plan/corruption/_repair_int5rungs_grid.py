"""Repair rust_grid.py: the previous insert split the `l2i6s4n1i8s2` entry in half.

The insert was placed after the FIRST line of a 3-line dict entry, so the entry's remaining two lines
became part of the inserted block. Rewrite the whole broken range as four correct entries.
"""
from pathlib import Path

P = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\rust_grid.py")
raw = P.read_bytes()

START = b'    "l2i6s4n1i8s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int8",'
END = b'"RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},'

i = raw.index(START)
j = raw.index(END, i) + len(END)
print("range", i, j, "len", j - i)

EOL = b"\r\n"
block = EOL.join([
    b'    "l2i6s4n1i8s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int8",',
    b'                     "RWKV_QUANT_SHIFTS": "1",',
    b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},',
    b'',
    b'    # THE int5/int4 CARD RUNGS UNDER THE NOTE\'s int3 SHIFT RUNG (2026-11-22). Chain 21 tests the',
    b'    # int6 card at 3,136 bits (1,024 + 2,112). The int5 card buys 128 more bits off the SAME note',
    b'    # (896 + 2,112 = 3,008) and has never been tried; the int4 card buys another 128 (768 + 2,112 =',
    b'    # 2,880) but carried +0.000214/+0.000390 on the tail at int4 SHIFTS (row 23). The third entry is',
    b'    # the int5 card under the note\'s int2 shift rung (896 + 1,920 = 2,816, 128 below chain 21\'s',
    b'    # gated 2,944). These three price the whole sub-3,136 corner on the CPU before any GPU run.',
    b'    "l2i5s4n1i8s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int5,note:1:int8",',
    b'                     "RWKV_QUANT_SHIFTS": "1",',
    b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},',
    b'    "l2i4s4n1i8s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:1:int8",',
    b'                     "RWKV_QUANT_SHIFTS": "1",',
    b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},',
    b'    "l2i5s4n1i8s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int5,note:1:int8",',
    b'                     "RWKV_QUANT_SHIFTS": "1",',
    b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},',
])

raw = raw[:i] + block + raw[j:]
P.write_bytes(raw)

import sys
sys.path.insert(0, str(P.parent))
import importlib

import rust_grid as rg

importlib.reload(rg)
for name in ("l2i6s4n1i8s2", "l2i6s4n1i8s3", "l2i5s4n1i8s3", "l2i4s4n1i8s3", "l2i5s4n1i8s2"):
    entry = rg.CONFIGS[name]
    bad = [k for k in entry if not k.startswith("RWKV_")]
    print(f"  {name:16s} keys={len(entry)} bogus={bad} bits={rg.bits_for(entry)}")
print("CONFIGS size", len(rg.CONFIGS))
