"""Remove the duplicated tail block in optimization/GPU_RUN_BOARD.md.

The 2026-11-19 22:35 'Row 11 amendment 2' block was appended twice (the one-shot helper ran
twice). Keep the first occurrence, drop the later identical copy. The board has a BOM/special
characters, so `edit` cannot match it -- hence this script.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "optimization" / "GPU_RUN_BOARD.md"

text = P.read_text(encoding="utf-8")
marker = "Row 11 amendment 2 (2026-11-19 22:35, Moscow)."
n = text.count(marker)
print("marker occurrences:", n)
if n <= 1:
    print("nothing to do")
    raise SystemExit(0)

first = text.find(marker)
second = text.find(marker, first + 1)
removed = text[first:second]
print("removing chars:", len(removed), "lines:", removed.count("\n"))
text = text[:first] + text[second:]
text = text.rstrip("\n") + "\n"
P.write_text(text, encoding="utf-8")
print("lines now:", len(text.splitlines()))
print("marker occurrences after:", text.count(marker))
