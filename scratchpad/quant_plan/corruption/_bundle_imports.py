"""Print the import lines of the row-17 harness scripts, to size the bundle's code closure."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = [
    "scratchpad/quant_plan/qat_decay.py",
    "scratchpad/quant_plan/quant_sweep.py",
    "scratchpad/quant_plan/it36ladder_lib.py",
    "scratchpad/adaptive_wd/adaptive_train.py",
    "scratchpad/quant_plan/rust_grid.py",
    "scratchpad/quant_plan/rust_speed.py",
    "scratchpad/quant_plan/rust_score.py",
    "scratchpad/quant_plan/rust_finite.py",
    "scratchpad/quant_plan/corruption/check_corruption.py",
    "scratchpad/quant_plan/corruption/finite_probe.py",
    "scratchpad/quant_plan/corruption/probe_get_result.py",
    "scratchpad/quant_plan/export_weights_only.py",
    "export_rnn_trace.py",
]
for f in FILES:
    p = ROOT / f
    if not p.exists():
        print(f"MISSING {f}")
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    imps = [l.strip() for l in text.splitlines() if re.match(r"^(import|from) ", l)]
    print(f"== {f}")
    for i in imps:
        print("   ", i)
