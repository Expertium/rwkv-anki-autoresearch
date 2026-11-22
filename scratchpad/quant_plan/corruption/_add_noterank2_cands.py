"""Insert the rank-2-note candidates into quant_sweep.py's CONFIGS, after the `l3i8s4n4s2` entry."""
from pathlib import Path

SWEEP = Path(__file__).resolve().parents[1] / "quant_sweep.py"
text = SWEEP.open(encoding="utf-8", newline="").read()

if '"l2i8n2i8"' in text:
    raise SystemExit("candidates already present - nothing to do")

BLOCK = '''    # Andrew's 2026-11-22 12:01 direction: put RANK 2 on the NOTE at int8 factors. The note is the big
    # pool (2/3 of a passing row's bits) and it is the one stream whose truncation tax nobody has cut
    # with precision - rows 19-22, 26 and the shrink combos all moved the note DOWN in bit width and
    # failed, while the card's ladder shows rank is what buys a truncated stream. `n2i4s4` (note rank 2
    # int4 at the SAME 2,304 bits as row 17) failed by a hair on the imm head (ub95 +0.000543), so the
    # untried point is rank 2 at int8: it SPENDS note bits (2,304 -> 3,840) instead of keeping them.
    # The note bar is 6,528 bits, so rank 2 (3,840) and even rank 3 (5,376) fit with room to spare.
    # `l2i8n2i8` is the direct A/B against row 17: same card, only the note's rank changes.
    "l2i8n2i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:2:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "card rank-2 int8 + int4 shifts, note rank-2 int8 + int4 (1,280/3,840 = 5,120)", None),
    "l3i8n2i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:2:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "row 18's card (rank-3 int8) under the note rank-2 int8 (1,792/3,840 = 5,632)", None),
    "l2i8n2i8s8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:2:int8",
                    "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int8"},
                   "note rank-2 int8 with int8 shifts: is the note's SHIFT precision free at rank 2? (1,280/4,608)", None),
    "l2i8n3i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:3:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "note rank-3 int8: the rank ladder on the note, 5,376 of the 6,528 note bits", None),
'''

anchor = '    "l3i8s4n4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",'
idx = text.find(anchor)
if idx < 0:
    raise SystemExit("anchor not found")
end = text.find('\n    "l2i8":', idx)
if end < 0:
    raise SystemExit("end of the anchor entry not found")

block = BLOCK.strip("\n").replace("\n", "\r\n") + "\r\n"
text = text[:end] + "\r\n" + block + text[end + 1:]

with SWEEP.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("inserted 4 candidates")
