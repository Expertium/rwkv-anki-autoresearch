"""Add the two chain-18 tags to record_corruption.ROWS so their corruption cells can be filled.

WHY. `find_row` first looks up the tag in the explicit ROWS map and only then falls back to "exactly one
row-line mentions the tag". Rows 41 and 45 pre-register the two mixed int6 points by name in their own
cells (row 41 prices the int6 rank-2 card as `l2i6c`; row 45 quotes the note cut's PTQ price as `l2i6n`),
so both tags now appear on TWO row-lines and the fallback resolves neither - `record_corruption.py`
printed "no unique RESULTS.md row yet" for both, although rows 48 and 49 exist and both checks PASSed.

The map is exactly the mechanism for this case ("the explicit ROWS map below"), so the fix is two entries
and no change to the matcher. Fails loudly if the anchor is not unique, if a tag is already mapped, or if
the target row number does not resolve to exactly one row-line.
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

P = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\corruption\record_corruption.py")
RESULTS = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\RESULTS.md")

ANCHOR = '    "l2i4s4": "23",\n'
ADD = ('    "l2i4s4": "23",\n'
       '    # Added 2026-11-23: rows whose Method cell is not the only line naming the tag (rows 41 and\n'
       '    # 45 pre-register the mixed int6 points by name), so the "exactly one row-line mentions it"\n'
       '    # fallback cannot resolve them.\n'
       '    "l2i6c": "48",     # chain 18 point 1 (card rank-2 int6 + note rank-1 int8, 3,328 bits)\n'
       '    "l2i6n": "49",     # chain 18 point 2 (card rank-2 int8 + note rank-1 int6, 3,200 bits)\n')

raw = P.read_bytes()
text = raw.decode("utf-8")
crlf = text.count("\r\n") > text.count("\n") - text.count("\r\n")
lf = text.replace("\r\n", "\n")
print(f"record_corruption.py: {len(raw)} bytes, {'CRLF' if crlf else 'LF'}")

if '"l2i6c": "48"' in lf:
    raise SystemExit("FAIL: l2i6c already mapped")
if lf.count(ANCHOR) != 1:
    raise SystemExit(f"FAIL: anchor x{lf.count(ANCHOR)}")

# the target rows must resolve to exactly one row-line each, or the entry is useless
lines = RESULTS.read_bytes().decode("utf-8").split("\n")
for num in ("48", "49"):
    pat = re.compile(rf"^\|\s*{num}\s*\|")
    hits = [i for i, l in enumerate(lines) if pat.match(l)]
    print(f"  RESULTS.md row {num}: {len(hits)} row-line(s)")
    if len(hits) != 1:
        raise SystemExit(f"FAIL: row {num} does not resolve uniquely")

out = lf.replace(ANCHOR, ADD, 1)
data = out.replace("\n", "\r\n") if crlf else out
P.write_bytes(data.encode("utf-8"))
if P.read_bytes().decode("utf-8") != data:
    raise SystemExit("FAIL: round trip differs")
print(f"record_corruption.py: written ({len(raw)} -> {len(data.encode('utf-8'))} bytes)")

import ast  # noqa: E402
ast.parse(P.read_text(encoding="utf-8"))
print("syntax ok")
