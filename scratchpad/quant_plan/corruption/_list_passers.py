import json
import pathlib

RUN = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan")
for stamp in ("it36l4i4s4", "it36l7i4s4", "it36n2i4s4", "it36l2i8496", "it36l2i8992"):
    p = RUN / f"quant_sweep_{stamp}.json"
    if not p.exists():
        print(f"{stamp:14s} no sweep json")
        continue
    r = json.loads(p.read_text(encoding="utf-8"))["rows"][0]
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    ok = a["ub95"] <= 0.0005 and i["ub95"] <= 0.0005
    print(f"{stamp:14s} card={r['bits']['card']:5d} note={r['bits']['note']:5d} "
          f"ahead {a['delta']:+.6f} ub95 {a['ub95']:+.6f} | imm {i['delta']:+.6f} ub95 {i['ub95']:+.6f}"
          f" -> {'PASS' if ok else 'FAIL'}")