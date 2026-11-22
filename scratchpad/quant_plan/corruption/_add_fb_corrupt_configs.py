"""Register the two fixed-basis rows in check_corruption.CONFIGS so a corruption check can run on them
the moment their sweep passes (Andrew's standing order: never leave a corruption check pending).

`run_one` takes the quantization env from quant_sweep.CANDIDATES[cand], so the middle element must be
the candidate name (identical to the config name here).
"""
from pathlib import Path

CC = Path(__file__).resolve().parents[3] / "scratchpad" / "quant_plan" / "corruption" / "check_corruption.py"

ANCHOR = '''    "l2i8n3i8": (RUN / "qat_it36l2i8n3i8" / "qat_124.pth", "l2i8n3i8", "chain 14 point 4"),
}'''

NEW = '''    "l2i8n3i8": (RUN / "qat_it36l2i8n3i8" / "qat_124.pth", "l2i8n3i8", "chain 14 point 4"),
    # FIXED-BASIS (codec) rows, chain 19 (2026-11-22): the note holds row 17's rank-1 payload size at
    # m=8 and beats it at m=6. Registered here so the check is one command once a sweep passes.
    "l2i8fb8n": (RUN / "qat_it36l2i8fb8n" / "qat_124.pth", "l2i8fb8n",
                 "chain 19 point 1 (fixed-basis note m=8, row 17's exact size)"),
    "l2i8fb6n": (RUN / "qat_it36l2i8fb6n" / "qat_124.pth", "l2i8fb6n",
                 "chain 19 point 2 (fixed-basis note m=6, 672 bits below row 17)"),
}'''

raw = CC.read_text(encoding="utf-8")
if '"l2i8fb8n"' in raw:
    raise SystemExit("check_corruption.py already has the fb rows")
n = raw.count(ANCHOR)
if n != 1:
    raise SystemExit(f"anchor matched {n} times (need 1)")
CC.write_text(raw.replace(ANCHOR, NEW, 1), encoding="utf-8")
print(f"patched {CC}")
