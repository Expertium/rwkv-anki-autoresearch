"""Add the three untested rungs below chain 21's 3,136-bit point to rust_grid.CONFIGS.

  l2i5s4n1i8s3  card rank-2 int5 + int4 shifts (896), note rank-1 int8 + int3 shifts (2,112) = 3,008
  l2i4s4n1i8s3  card rank-2 int4 + int4 shifts (768), note rank-1 int8 + int3 shifts (2,112) = 2,880
  l2i5s4n1i8s2  card rank-2 int5 + int4 shifts (896), note rank-1 int8 + int2 shifts (1,920) = 2,816

The note's int3 shift rung is row 35's (`n1i8s3`, +0.000063/+0.000364) and its int2 rung is chain 21's
gated point. The only new ingredient is the int5 card factor width, which the engine supports
(model.rs parse_level) and which the Python QAT path gained on 2026-11-22.
"""
from pathlib import Path

P = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\rust_grid.py")
raw = P.read_bytes()

ANCHOR_LF = b'    "l2i6s4n1i8s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int8",'
if ANCHOR_LF + b"\r\n" in raw:
    eol = b"\r\n"
elif ANCHOR_LF + b"\n" in raw:
    eol = b"\n"
else:
    raise SystemExit("ABORT: anchor line not found")

if b'"l2i5s4n1i8s3"' in raw:
    print("rust_grid.py: entries already present")
else:
    lines = [
        b"",
        b"    # THE int5 CARD RUNGS (2026-11-22): chain 21 tests the int6 card under the note's int3 shift",
        b"    # rung (3,136). The int5 card buys 128 more bits off the SAME note and has never been tried;",
        b"    # the int4 card buys 256 but carried +0.000214/+0.000390 at int4 SHIFTS (row 23) and the",
        b"    # int2-shift note rung is a coin flip (row 20), so these three price the whole sub-3,136 corner.",
        b'    "l2i5s4n1i8s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int5,note:1:int8",',
        b'                     "RWKV_QUANT_SHIFTS": "1",',
        b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},',
        b'    "l2i4s4n1i8s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:1:int8",',
        b'                     "RWKV_QUANT_SHIFTS": "1",',
        b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},',
        b'    "l2i5s4n1i8s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int5,note:1:int8",',
        b'                     "RWKV_QUANT_SHIFTS": "1",',
        b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},',
    ]
    ins = eol.join(lines) + eol
    i = raw.index(ANCHOR_LF + eol) + len(ANCHOR_LF + eol)
    raw = raw[:i] + ins + raw[i:]
    P.write_bytes(raw)
    print("rust_grid.py: 3 entries added")

# verify the registry loads and the bit counts are what the comment claims
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("rg", P)
rg = importlib.util.module_from_spec(spec)
sys.modules["rg"] = rg
spec.loader.exec_module(rg)
for name in ("l2i6s4n1i8s3", "l2i5s4n1i8s3", "l2i4s4n1i8s3", "l2i5s4n1i8s2"):
    b = rg.bits_for(rg.CONFIGS[name])
    print(f"  {name:16s} {b}  total {sum(b.values())}")
print("done")
