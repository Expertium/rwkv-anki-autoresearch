"""Add the NOTE-ONLY fixed-basis candidates to quant_sweep.CANDIDATES.

WHY THE NOTE ONLY (corruption/fixedbasis_l2i8s4all.json, leave-one-user-out): the card is reproduced
EXACTLY at rank 2 for 1,280 bits, while the best fixed basis on the card needs m=16 (2,304 bits) to
reach only 0.9596 held out. The note's shipped rank-1 captures 0.904-0.951 and passes the gate, and a
fixed basis reaches 0.876-0.921 held out at m=8 for the SAME 768 bits/layer (0.82-0.87 at m=6, 544
bits/layer). So: hold the card at rank 2, switch only the note.

  l2i8fb8n  card 1,280 + note 3x768 = 3,584  (row 17's exact size -> a pure codec A/B)
  l2i8fb6n  card 1,280 + note 3x544 = 2,912  (672 below row 17 -> the size claim)

The note's m comes from the QAT lowrank scope's rank slot; RWKV_FIXEDBASIS_SCOPE=note keeps the card
on the rank-r path (rwkv/model/rwkv_ops.py, 2026-11-22).
"""
from pathlib import Path

QS = Path(__file__).resolve().parents[3] / "scratchpad" / "quant_plan" / "quant_sweep.py"
BASIS = ("C:\\\\Users\\\\Andrew\\\\rwkv-anki-autoresearch\\\\scratchpad\\\\quant_plan\\\\corruption"
         "\\\\fixedbasis_l2i8s4all.npz")

ANCHOR = '''    "fb4i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:4:int8,note:4:int8",'''

NEW = '''    # THE NOTE-ONLY FIXED-BASIS CANDIDATES (2026-11-22, after the leave-one-user-out measurement).
    # WHY THE NOTE ONLY: at rank 2 the CARD is reproduced EXACTLY for 1,280 bits, and the best fixed
    # basis on the card needs m=16 (2,304 bits) to reach only 0.9596 held out - the card is the wrong
    # place for a codec. The NOTE's shipped rank-1 captures 0.904-0.951 and passes the gate, and a
    # fixed basis reaches 0.876-0.921 held out at m=8 for the SAME 768 bits/layer (0.82-0.87 at m=6,
    # 544 bits/layer). So these hold the card at rank 2 and switch only the note (RWKV_FIXEDBASIS_SCOPE):
    #   l2i8fb8n  card 1,280 + note 3x768 = 3,584  (row 17's EXACT size -> a pure codec A/B)
    #   l2i8fb6n  card 1,280 + note 3x544 = 2,912  (672 below row 17 -> the size claim)
    "l2i8fb8n": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:8:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                  "RWKV_FIXEDBASIS_SCOPE": "note",
                  "RWKV_FIXEDBASIS_PATH": "BASISFILE"},
                 "card rank 2 (row 17's card) + NOTE fixed basis m=8 int8 (1,280/2,304 = 3,584 bits)",
                 (1280, 2304)),
    "l2i8fb6n": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:6:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                  "RWKV_FIXEDBASIS_SCOPE": "note",
                  "RWKV_FIXEDBASIS_PATH": "BASISFILE"},
                 "card rank 2 (row 17's card) + NOTE fixed basis m=6 int8 (1,280/1,632 = 2,912 bits)",
                 (1280, 1632)),
    "fb4i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:4:int8,note:4:int8",'''

raw = QS.read_text(encoding="utf-8")
if '"l2i8fb8n"' in raw:
    raise SystemExit("quant_sweep.py already has the note-only fb candidates")
n = raw.count(ANCHOR)
if n != 1:
    raise SystemExit(f"fb4i8 anchor matched {n} times (need 1)")
raw = raw.replace(ANCHOR, NEW.replace("BASISFILE", BASIS), 1)
QS.write_text(raw, encoding="utf-8")
print(f"patched {QS} ({len(raw)} bytes)")
