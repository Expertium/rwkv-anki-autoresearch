import re
from pathlib import Path

RUN = Path("scratchpad/quant_plan")
for p in sorted(RUN.glob("it36*_chain.py")):
    s = p.read_text(encoding="utf-8", errors="replace")
    prev = re.findall(r'PREV_END\s*=\s*"([^"]+)"', s)
    prevl = re.findall(r'PREV_LOG\s*=\s*RUN\s*/\s*"([^"]+)"', s)
    wait = re.findall(r'WAIT_MAX_MIN\s*=\s*(\d+)', s)
    points = re.findall(r'^\s*\(\s*"([A-Za-z0-9_]+)",\s*"it36', s, re.M)
    print(f"{p.name:26s} prev={prev} prevlog={prevl} wait={wait} pts={points}")
