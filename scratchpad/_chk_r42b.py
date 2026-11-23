from pathlib import Path

RUN = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan")
s = (RUN / "RESULTS.md").read_text(encoding="utf-8")

rows = [l for l in s.splitlines() if l.startswith("| 4") ]
for l in rows:
    cells = l.split("|")
    print(len(cells), "cells |", l[:200])
    print("...tail:", l[-220:])
    print("===")

import re
m = [l for l in s.splitlines() if "rows 33-42 FAIL too" in l]
print("STATUS LEN", len(m), m[0][:400] if m else "")
