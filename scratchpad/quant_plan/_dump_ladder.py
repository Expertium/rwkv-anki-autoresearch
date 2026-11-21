import json
import pathlib

for s in ("it36l3i8", "it36l2i8s4", "it36l3i8s4", "it36nr4", "it36nr4s2"):
    p = pathlib.Path("scratchpad/quant_plan/quant_sweep_" + s + ".json")
    if not p.exists():
        print(f"{s:14s} -- not yet")
        continue
    r = json.loads(p.read_text())["rows"][0]
    a, i = r["gate"]["ahead"], r["gate"]["imm"]
    print(f"{s:14s} {r['bits']} ahead {a['mean_cur']:.6f} d {a['delta']:+.6f} ub95 {a['ub95']:+.6f} | "
          f"imm {i['mean_cur']:.6f} d {i['delta']:+.6f} ub95 {i['ub95']:+.6f}")
