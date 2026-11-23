import re
from pathlib import Path

RUN = Path("scratchpad/quant_plan")
s = (RUN / "RESULTS.md").read_text(encoding="utf-8")
lines = s.splitlines()
print("total lines", len(lines))
for l in lines:
    if l.startswith("| 23 |") or l.startswith("| 41 |") or l.startswith("| 42 |"):
        print(l[:300])
        print("  cells:", len(l.split("|")))
        print("---")
print("STATUS:")
for l in lines:
    if l.startswith("**Status:"):
        print(l[:600])
print()
print("=== wide_screen.py head")
w = (RUN / "corruption" / "wide_screen.py").read_text(encoding="utf-8")
print(w[:1800])
