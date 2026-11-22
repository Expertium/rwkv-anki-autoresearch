"""Repair the BASIS_PATH placeholder collision: my replace() also hit the RWKV_FIXEDBASIS_PATH key
name, leaving `"RWKV_FIXED"scratch/...txt"": ` as a key. Restore it to the env-var name."""
from pathlib import Path

GRID = Path(__file__).resolve().parents[3] / "scratchpad" / "quant_plan" / "rust_grid.py"
BAD = '"RWKV_FIXED"scratchpad/quant_plan/corruption/fixedbasis_l2i8s4all.txt"'
GOOD = '"RWKV_FIXEDBASIS_PATH"'
raw = GRID.read_text(encoding="utf-8")
n = raw.count(BAD)
if n == 0:
    raise SystemExit("no broken key found; nothing to repair")
raw = raw.replace(BAD, GOOD)
GRID.write_text(raw, encoding="utf-8")
print(f"repaired {n} keys ({len(raw)} bytes)")
