import re

s = open("scratchpad/quant_plan/quant_sweep.py", encoding="utf-8").read()
for name in ("l2i5s4", "l2i4s4", "l2i6c", "l2i6s4n1i8s3", "l2i6s4n1i8s2"):
    m = re.search(r'\n    "' + re.escape(name) + r'": \(', s)
    if not m:
        print(f"{name}: ABSENT")
        continue
    print(s[m.start():m.start() + 380])
    print("---")
print("## lines mentioning QMAX / int5 / int6")
for l in s.splitlines():
    if "_QMAX" in l or "int5" in l or "int6" in l:
        print("   ", l.strip()[:160])
