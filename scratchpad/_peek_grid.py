import json
import re

s = open("scratchpad/quant_plan/rust_grid.py", encoding="utf-8").read()
m = re.search(r"CONFIGS = \{.*?\n\}", s, re.S)
print(m.group(0))
print("=" * 60)
d = json.load(open("scratchpad/quant_plan/corruption/state_subspace_l2i8s4all.json", encoding="utf-8"))
for k, v in d.items():
    if isinstance(v, dict):
        svd = v.get("svd")
        print(k, "n=", v.get("n"), "energy=", v.get("energy"), "svd=", svd)
