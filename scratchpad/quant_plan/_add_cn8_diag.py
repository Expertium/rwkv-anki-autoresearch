"""One-shot: add the over-the-bar full-matrix int8 diagnostic `cn8` to quant_sweep.CANDIDATES.

`edit` cannot match quant_sweep.py (CRLF), so the insertion happens here with CRLF preserved.

Why: row 4 (rank-1 int8, 1,024 bits/layer) leaves the immediate head +0.001009, i.e. 2x over the
ceiling, and we cannot tell whether that error is the rank-1 TRUNCATION (the WKV matrix is only
approximately rank 1) or the FACTOR PRECISION (int8). `cn8` quantizes both streams per element at int8
with no truncation at all: 8,704 card bits / 26,112 note bits, four times the constraint-2 bars, so it
is a DIAGNOSTIC ONLY and never a candidate. If its tax is near zero, the truncation is the whole
problem and a finer level cannot fix it; if its tax is still ~+0.001, precision is not the mechanism.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "quant_sweep.py"
with P.open(encoding="utf-8", newline="") as fh:   # newline="" keeps the CRLF endings intact
    text = fh.read()
nl = "\r\n"

anchor = ('    "cn3": ({"RWKV_QAT_SCOPE": "card:int3,note:int3", "RWKV_QAT_SHIFT_SCOPE": "card:int3,note:int3"},')
if text.count(anchor) != 1:
    raise SystemExit(f"anchor found {text.count(anchor)} times")

block = ('    # DIAGNOSTIC, far over the bar: no rank truncation at all, 8 bits per state element. It tells\n'
         '    # us whether the residual immediate-head tax is rank-1 truncation or factor precision.\n'
         '    "cn8": ({"RWKV_QAT_SCOPE": "card:int8,note:int8", "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},\n'
         '            "DIAGNOSTIC, 4x over the bar: full-matrix int8 both streams (no rank truncation)", None),\n')

text = text.replace(anchor, block + anchor)
P.write_text(text, encoding="utf-8", newline="")

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("qs", P)
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)
for name in ("cn8", "l1i8", "cn2"):
    env, comment, override = qs.CANDIDATES[name]
    print(f"{name:6s} {qs.bits_for(env, override)}  {comment}")
print("candidates:", len(qs.CANDIDATES))
