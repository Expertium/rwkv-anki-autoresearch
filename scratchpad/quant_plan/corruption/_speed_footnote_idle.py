"""Rewrite the rank-2 speed paragraph in RESULTS.md with the finished idle-machine re-measurement.

The file has mixed line endings, so match with whitespace-tolerant regexes, not literal blocks.
"""
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "RESULTS.md"  # scratchpad/quant_plan/RESULTS.md

IDLE = ("and the idle-machine re-measurement settles it: `scratchpad/quant_plan/"
        "rust_speed_idle_l2i8.json` (2026-11-22 11:47, `it36l2i8_124.safetensors`, users 107/136/156, "
        "2 reps, GPU idle) reads **2,357.8 rev/s = 0.809x** for the rank-2 card path (states/s 0.658x, "
        "median per-user 0.795x) against plain 2,915.1 rev/s, so the under-load figure was accurate to "
        "about 1% rather than 3% pessimistic.")

TAIL = ("(`l1i8` reads 0.902x on that idle run, matching the 0.903x it read in the other idle "
        "measurement). Every config of that run passes the floor - plain 1.000x, `l1i8` 0.902x, "
        "`l2i8` 0.809x, `l2i8s4` 0.809x, `l3i8s4` 0.809x - and the three rank>=2 card configs are "
        "indistinguishable from each other, because neither the factor width nor the shift width "
        "changes the card's verify path: only the rank does. The constraint-6 verdicts of rows 15-18 "
        "stand unchanged.")

text = RESULTS.open(encoding="utf-8", newline="").read()
before = text

text, n1 = re.subn(r"so it is about 3% pessimistic", IDLE.replace("\\", "\\\\"), text, count=1)
text, n2 = re.subn(r"\(`l1i8` reads 0\.875x.*?still owed\.", TAIL.replace("\\", "\\\\"), text,
                   count=1, flags=re.S)
if n1 != 1 or n2 != 1:
    raise SystemExit(f"expected one match each, got {n1} and {n2}")

with RESULTS.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print(f"patched ({len(before)} -> {len(text)} chars)")
