import json
import os
from pathlib import Path

RUN = Path("scratchpad/quant_plan")
for name in ["rust_wide_screen_it36fbwide_b1.json", "rust_grid_it36fbwide_b1.json",
             "rust_wide_screen_it36fbwide_b1_it36int6shift.json", "rust_grid_it36int6shift.json"]:
    p = RUN / name
    d = json.load(open(p, encoding="utf-8"))
    print("==", name, os.path.getsize(p))
    for k in ("tags", "base", "n_users", "users", "configs", "secs", "weights", "trace_dir", "meta"):
        if k in d:
            v = d[k]
            print("   ", k, "=", (list(v.keys()) if isinstance(v, dict) else str(v)[:200]))
    print("   keys:", list(d.keys()))
print("== rust_artifacts")
for f in sorted(Path("scratchpad/quant_plan/rust_artifacts").glob("*.safetensors")):
    print("   ", f.name, f.stat().st_size)
print("== traces")
t = sorted((Path("scratchpad/quant_plan/trace_corrupt50")).glob("trace_user_*.json"))
print("   n =", len(t))
print("   ", [int(p.stem.split("_")[-1]) for p in t])
