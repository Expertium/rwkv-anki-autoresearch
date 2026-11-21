import re
from pathlib import Path

with Path("scratchpad/quant_plan/quant_sweep.py").open(encoding="utf-8", newline="") as fh:
    s = fh.read()
names = re.findall(r'\n    "([^"]+)": \(', s)
dups = sorted({n for n in names if names.count(n) > 1})
print("entries:", len(names), "duplicates:", dups)
start = s.find("CANDIDATES = {")
end = s.find("\n}\n", start)
block = s[start:end]
print("names in CANDIDATES block:", len(re.findall(r'\n    "([^"]+)": \(', block)))
for n in ("n4", "n4s2"):
    for m in re.finditer(r'\n    "' + n + r'": \(', s):
        print("---", n, "at", m.start())
        print(s[m.start():m.start() + 320].replace("\r\n", "\n"))
