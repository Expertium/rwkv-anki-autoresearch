"""Insert the equal-bits (rank, precision) probes and the aggressive shrink combos.

Row 15 showed RANK is what buys the immediate head (rank-1 -> rank-2: +0.001009 -> +0.000307), and `cn8`
showed 8-bit precision of the FULL state is free. That makes the (rank, precision) split at a FIXED bit
count the next question, because the two sides cost the same bits:

  card 768 bits   = rank-1 int8 + int4 shifts   (row 9)   vs rank-2 int4 + int4 shifts
  card 1,280 bits = rank-2 int8 + int4 shifts   (row 17)  vs rank-4 int4 + int4 shifts
  card 2,048 bits = rank-3 int8 + int8 shifts   (row 16)  vs rank-7 int4 + int4 shifts
  note 2,304 bits = rank-1 int8 + int4 shifts   (row 17)  vs rank-2 int4 + int4 shifts

and the aggressive shrink combos push the total state below row 15's 4,608 bits.
"""
from pathlib import Path

P = Path("scratchpad/quant_plan/quant_sweep.py")
with P.open(encoding="utf-8", newline="") as fh:  # newline="" keeps the CRLF endings intact
    text = fh.read()

anchor = '    "l2i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",\r\n'
if text.count(anchor) != 1:
    raise SystemExit(f"anchor found {text.count(anchor)} times")

block = (
    "    # Equal-bits (rank, precision) probes. Rank is what buys the immediate head and 8-bit precision\r\n"
    "    # of the full state is free, so at a FIXED bit count the question is rank vs precision. The note\r\n"
    "    # is held at rank-1 int8 + int4 shifts (2,304 bits) except in `n2i4s4`, which makes the same\r\n"
    "    # comparison on the note side. Added 2026-11-21.\r\n"
    '    "l2i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int4,note:1:int8",\r\n'
    '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '               "card rank-2 int4 + int4 shifts, 768 card bits (vs row 9 rank-1 int8)", None),\r\n'
    '    "l4i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:4:int4,note:1:int8",\r\n'
    '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '               "card rank-4 int4 + int4 shifts, 1,280 card bits (vs row 17 rank-2 int8)", None),\r\n'
    '    "l7i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:7:int4,note:1:int8",\r\n'
    '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '               "card rank-7 int4 + int4 shifts, 2,048 card bits (vs row 16 rank-3 int8)", None),\r\n'
    '    "n2i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:2:int4",\r\n'
    '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '               "note rank-2 int4 + int4 shifts, 2,304 note bits (vs row 17 note rank-1 int8)", None),\r\n'
    "    # Aggressive shrink combos: push the TOTAL below row 15's 4,608 bits.\r\n"
    '    "l3i4s4n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int4,note:1:int4",\r\n'
    '                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '                 "card rank-3 int4 + int4 shifts, note rank-1 int4 + int4 (1,024/1,536)", None),\r\n'
    '    "l2i8s4n3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int3",\r\n'
    '                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '                 "card rank-2 int8 + int4 shifts, note rank-1 int3 + int4 (1,280/1,344)", None),\r\n'
    '    "l2i8s2n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int4",\r\n'
    '                  "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int4"},\r\n'
    '                 "card rank-2 int8 + int2 shifts, note rank-1 int4 + int4 (1,152/1,536)", None),\r\n'
    '    "l3i8s4n4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",\r\n'
    '                    "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int2"},\r\n'
    '                   "card rank-3 int8 + int4 shifts, note rank-1 int4 + int2 (1,792/1,152)", None),\r\n'
)
text = text.replace(anchor, block + anchor, 1)
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("inserted")
