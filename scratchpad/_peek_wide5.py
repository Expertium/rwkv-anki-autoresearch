import os
from pathlib import Path

root = Path("scratchpad/quant_plan/rust_grid_preds/it36wide5")
if root.exists():
    for d in sorted(root.iterdir()):
        files = sorted(p.name for p in d.iterdir())
        print(f"{d.name:16s} {len(files)}  {files}")
else:
    print("no preds root yet")
log = Path("scratchpad/quant_plan/rust_grid_it36wide5_stdout.log")
print("--- stdout log", log.stat().st_size if log.exists() else "absent")
if log.exists():
    print(log.read_text(encoding="utf-8", errors="replace"))
