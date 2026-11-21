"""Insert the card rank-2/3 candidates into quant_sweep.CANDIDATES (CRLF preserved).

Why: `cn8` (full-matrix int8, no rank truncation) scored +0.000033 ahead / +0.000022 imm against the
fp32 control, i.e. 8-bit precision of the FULL state is essentially free. The rank-1 tax
(+0.000503/+0.001009) is therefore TRUNCATION, not precision. The attribution pair says the card stream
owns it (card-only +0.001142 imm vs note-only +0.000258). So the next lever is the card's rank, and the
card bar (2,176 bits) has room: rank-2 int8 + int4 shifts = 1,280 bits and even rank-3 int8 + int8
shifts = 2,048 bits both fit.
"""
from pathlib import Path

P = Path("scratchpad/quant_plan/quant_sweep.py")
with P.open(encoding="utf-8", newline="") as fh:  # newline="" keeps the CRLF endings intact
    text = fh.read()

anchor = '    "l1i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int8",\r\n'
if text.count(anchor) != 1:
    raise SystemExit(f"anchor found {text.count(anchor)} times")

nl = "\r\n"
block = (
    "    # Card rank-r. `cn8` (no truncation at all, 8 bits) has a ~zero tax (+0.000033/+0.000022 vs\r\n"
    "    # the fp32 control), so 8-bit precision of the full state is free and the rank-1 tax is\r\n"
    "    # TRUNCATION. The per-stream attribution says the card owns it (card-only +0.001142 imm vs\r\n"
    "    # note-only +0.000258), and the card's 2,176-bit bar has room for more ranks. The note stays\r\n"
    "    # rank-1 int8 in all four. Added 2026-11-21.\r\n"
    '    "l2i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",\r\n'
    '              "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},\r\n'
    '             "card rank-2 int8 factors + int8 shifts (1,536/3,072 bits)", None),\r\n'
    '    "l3i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int8",\r\n'
    '              "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},\r\n'
    '             "card rank-3 int8 factors + int8 shifts (2,048/3,072 bits)", None),\r\n'
    '    "l2i8s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",\r\n'
    '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '               "card rank-2 int8 factors + int4 shifts (1,280/2,304 bits)", None),\r\n'
    '    "l3i8s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int8",\r\n'
    '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '               "card rank-3 int8 factors + int4 shifts (1,792/2,304 bits)", None),\r\n'
)
text = text.replace(anchor, block + anchor, 1)
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("inserted; CANDIDATES count:", text.count('": ({') + text.count('": ({}'))
