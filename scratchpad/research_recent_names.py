from pathlib import Path
import re
s=Path("optimization/research_5k.md").read_text(encoding="utf-8",errors="replace")
for line in s.splitlines():
 if line.startswith("|") and "`" in line:
  first=line.split("|",2)[1].strip()
  m=re.search(r"(\d+)",first)
  if m and int(m.group(1))>=110: print(first)
