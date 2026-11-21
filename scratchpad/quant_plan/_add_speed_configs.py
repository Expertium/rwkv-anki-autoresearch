"""Add the CARD-rank-2 deploy configs to rust_grid.CONFIGS so rust_speed.py can measure them.

`l2i4` (rank-2 on BOTH streams) is the only rank-2 speed point and it fails: 0.629x rev/s, 0.448x
states/s. The winning loss recipe puts rank 2 on the CARD only (1 layer) and leaves the note at rank-1
(3 layers), and the rank-1 attribution says the card is the cheap stream (l1i4c 0.982x vs l1i4n 0.949x),
so the card-only rank-2 penalty should be much smaller than 0.629x.
"""
from pathlib import Path

P = Path("scratchpad/quant_plan/rust_grid.py")
text = P.read_text(encoding="utf-8")

anchor = '    "l2i4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:2:int4", "RWKV_QUANT_SHIFTS": "1"},\n'
assert text.count(anchor) == 1, text.count(anchor)

block = (
    '    # card-only rank-2: the shape the loss winner actually uses (RESULTS row 15). Added 2026-11-21.\n'
    '    "l2i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",\n'
    '             "RWKV_STATE_SHIFT_LEVEL": "int8"},\n'
    '    "l2i8s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",\n'
    '               "RWKV_STATE_SHIFT_LEVEL": "int4"},\n'
    '    "l3i8s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:3:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",\n'
    '               "RWKV_STATE_SHIFT_LEVEL": "int4"},\n'
)
text = text.replace(anchor, anchor + block, 1)
P.write_text(text, encoding="utf-8")
print("rust_grid.CONFIGS extended")
