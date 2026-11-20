"""One-shot: add the two 768/2,304-bit mixed-shift candidates to quant_sweep.CANDIDATES.

`edit` cannot match quant_sweep.py (CRLF), so the insertion is done here and the file is rewritten
with the same CRLF endings. Both candidates exist only for attribution: they have the SAME bit budget
as each other, with the bits moved from the rank-1 factors to the token-shift vectors or back, so the
pair says which side of the rank-1 split carries the quantization tax.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "quant_sweep.py"
with P.open(encoding="utf-8", newline="") as fh:   # newline="" keeps the CRLF endings intact
    text = fh.read()
nl = "\r\n"

anchor = ('    "l1i4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int4,note:1:int4",' + nl +
          '                "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int2"},' + nl +
          '               "rank-1 factors at int4, shifts int2 (smaller card state)", None),')
if text.count(anchor) != 1:
    raise SystemExit(f"anchor found {text.count(anchor)} times")

block = (nl +
         '    # Which side of the rank-1 split carries the tax? These two have the SAME state size' + nl +
         '    # (768 card / 2,304 note bits) with the bits moved from one side to the other, so the pair' + nl +
         '    # is an attribution test that costs no state. Added 2026-11-20 for the axis-1 curve.' + nl +
         '    "l1i8s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int8",' + nl +
         '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},' + nl +
         '               "rank-1 int8 factors + int4 shifts (768/2304 bits)", None),' + nl +
         '    "l1i4s8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int4,note:1:int4",' + nl +
         '                "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},' + nl +
         '               "rank-1 int4 factors + int8 shifts (768/2304 bits)", None),')

text = text.replace(anchor, anchor + block)
P.write_text(text, encoding="utf-8", newline="")

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("qs", P)
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)
for name in ("l1i8s4", "l1i4s8", "l1i4", "l1i4s2", "l1i8"):
    env, comment, override = qs.CANDIDATES[name]
    print(f"{name:8s} {qs.bits_for(env, override)}  {comment}")
print("candidates:", len(qs.CANDIDATES))
