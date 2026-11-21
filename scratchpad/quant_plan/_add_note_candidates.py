"""Insert the NOTE-precision and card-rank+note-int4 candidates into quant_sweep.CANDIDATES.

The note pool is 2/3 of the total bits (3,072 of 4,096 at row 4) while the note's own rank-1 int8 tax is
+0.000373/+0.000258 - i.e. inside the ceiling by itself. So the note is where the bits can be cut if its
int4 is cheap. `n4` isolates the note's int4 tax on row 4's exact card config; the `l*i*s4n4` points
combine the likely card fix (more rank) with the note cut.
"""
from pathlib import Path

P = Path("scratchpad/quant_plan/quant_sweep.py")
with P.open(encoding="utf-8", newline="") as fh:  # newline="" keeps the CRLF endings intact
    text = fh.read()

anchor = '    "l2i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",\r\n'
if text.count(anchor) != 1:
    raise SystemExit(f"anchor found {text.count(anchor)} times")

block = (
    "    # The note pool is 2/3 of the total bits (3,072 of 4,096 at row 4) and the note's OWN rank-1\r\n"
    "    # int8 tax is +0.000373/+0.000258 - inside the ceiling by itself. So the note is where the\r\n"
    "    # bits can be cut. `n4` isolates the note's int4 tax on row 4's exact card config; the last two\r\n"
    "    # combine the likely card fix (more rank) with the note cut. Added 2026-11-21.\r\n"
    '    "n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int4",\r\n'
    '            "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int4"},\r\n'
    '           "note rank-1 int4 factors + int4 shifts, card at row 4\'s exact config (1,024/1,536 bits)", None),\r\n'
    '    "n4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int4",\r\n'
    '              "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int2"},\r\n'
    '             "note rank-1 int4 factors + int2 shifts (1,024/1,152 bits)", None),\r\n'
    '    "l2i8s4n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int4",\r\n'
    '                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '                 "card rank-2 int8 + int4 shifts, note rank-1 int4 + int4 (1,280/1,536 bits)", None),\r\n'
    '    "l3i8s4n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",\r\n'
    '                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    '                 "card rank-3 int8 + int4 shifts, note rank-1 int4 + int4 (1,792/1,536 bits)", None),\r\n'
)
text = text.replace(anchor, block + anchor, 1)
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("inserted")
