import glob
import json
import os

WANT = {"7250", "6082", "5975", "5939", "6999", "7200", "5052"}

pats = ["scratchpad/quant_plan/*.json", "scratchpad/quant_plan/corruption/*.json"]
for pat in pats:
    for p in sorted(glob.glob(pat)):
        if os.path.getsize(p) > 60000:
            continue
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        hits = sum(1 for w in WANT if w in t)
        if hits == len(WANT):
            print(f"{hits}/{len(WANT)}  {p}  {os.path.getsize(p)}")
