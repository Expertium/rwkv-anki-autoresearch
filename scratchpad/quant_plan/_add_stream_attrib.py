"""One-shot: add the int8-only attribution candidates (`l1i8c`, `l1i8n`) to quant_sweep.CANDIDATES.

`edit` cannot match quant_sweep.py (CRLF), so the insertion happens here with CRLF preserved. They are
the per-stream twins of row 4's winning allocation: one quantizes only the card, the other only the
note, so the ahead/imm tax of row 4 splits into a card part and a note part at the exact best config.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "quant_sweep.py"
with P.open(encoding="utf-8", newline="") as fh:   # newline="" keeps the CRLF endings intact
    text = fh.read()
nl = "\r\n"

anchor = ('    "l1i8s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int8",')
if text.count(anchor) != 1:
    raise SystemExit(f"anchor found {text.count(anchor)} times")

block = ('    # Attribution for row 4 at its exact allocation: which stream owns which head? Card-only and\n'
         '    # note-only rank-1 int8, the other stream left in fp32. Added 2026-11-21.\n'
         '    "l1i8c": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8", "RWKV_QAT_SHIFT_SCOPE": "card:int8"},\n'
         '              "attribution: card rank-1 int8 only, note untouched", None),\n'
         '    "l1i8n": ({"RWKV_QAT_LOWRANK_SCOPE": "note:1:int8", "RWKV_QAT_SHIFT_SCOPE": "note:int8"},\n'
         '              "attribution: note rank-1 int8 only, card untouched", None),\n')

text = text.replace(anchor, block + anchor)
P.write_text(text, encoding="utf-8", newline="")

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("qs", P)
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)
for name in ("l1i8c", "l1i8n", "l1i8", "l1i8s4", "l1i8s4", "l1i4s8"):
    env, comment, override = qs.CANDIDATES[name]
    print(f"{name:8s} {qs.bits_for(env, override)}  {comment}")
print("candidates:", len(qs.CANDIDATES))
