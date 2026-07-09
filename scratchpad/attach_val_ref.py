"""One-off: attach champ5k_b1's WS validation trajectory to champion_5k.json as the vprune ref.

b1 predates the val-trace sidecar (train_rwkv now writes <trace>.val.jsonl when RWKV_STEP_TRACE is
on), so its val points are parsed from the run log: WS ran VALIDATE_EVERY=500 -> the FIRST 15
"Mean ahead validation loss" lines are the WS trajectory at steps [50, 500..6500, 6554] (the decay
phase adds exactly 2 more: VALIDATE_EVERY=100000 -> step 50 + final; 15+2=17 lines total, verified).
Future champions get val arrays natively via promote_champion_5k --val-trace. Scripted (not
hand-edited) per the no-hand-editing rule for champion_5k.json.
"""
import json
import re

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
LOG = f"{ROOT}/scratchpad/champ5k_b1/champ5k_b1.log"
CHAMP = f"{ROOT}/optimization/champion_5k.json"

pts = []
for line in open(LOG, encoding="utf-8", errors="replace"):
    m = re.match(r"Mean ahead validation loss: ([\d.]+) \([\d.]+\), imm: ([\d.]+)", line)
    if m:
        pts.append((float(m.group(1)), float(m.group(2))))
assert len(pts) == 17, f"expected 17 val lines (15 WS + 2 decay), got {len(pts)}"
ws = pts[:15]
steps = [50] + [500 * k for k in range(1, 14)] + [6554]
assert all(ws[i][0] >= ws[14][0] - 0.02 for i in range(1, 15)), "non-plausible trajectory"

champ = json.load(open(CHAMP))
assert champ["name"] == "champ5k_b1", champ["name"]
champ["val_step"] = steps
champ["val_ahead"] = [p[0] for p in ws]
champ["val_imm"] = [p[1] for p in ws]
champ["val_note"] = ("WS validation trajectory (VALIDATE_USERS 5001-5010, VALIDATE_EVERY=500), "
                     "parsed from champ5k_b1.log 2026-07-09; vprune reference")
with open(CHAMP, "w") as f:
    json.dump(champ, f)
print(f"attached {len(steps)} val points to {CHAMP}")
print(f"  first: step 50 = {ws[0]},  last: step 6554 = {ws[14]}")
