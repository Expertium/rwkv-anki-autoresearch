"""Rebuild + launch the TA3 0.04-epoch smoke after the per-group split fix (2026-11-16).

Reuses `scratchpad/adaptive_wd/mk_ta3_smoke_fixed.py` (same real start config = QAT46 step-248, same
env) with only the output folder and checkpoint prefix changed, so the 2026-11-15 smoke artifacts stay
intact for comparison. Same purpose as before: plumbing and cost check of the REAL code path, never
accuracy evidence.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIG = ROOT / "scratchpad" / "adaptive_wd" / "mk_ta3_smoke_fixed.py"
src = ORIG.read_text(encoding="utf-8")

OLD_DST = 'DST = ROOT / "scratchpad" / "ta3_smoke_fixed_20261115"'
NEW_DST = 'DST = ROOT / "scratchpad" / "ta3_smoke_fix2_20261116"'
assert src.count(OLD_DST) == 1
src = src.replace(OLD_DST, NEW_DST)

OLD_PREFIX = '"SAVE_MODEL_PREFIX": "ta3smoke",'
NEW_PREFIX = '"SAVE_MODEL_PREFIX": "ta3smoke2",'
assert src.count(OLD_PREFIX) == 1
src = src.replace(OLD_PREFIX, NEW_PREFIX)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exec(compile(src, str(ORIG), "exec"), {"__name__": "__main__", "__file__": str(ORIG)})
