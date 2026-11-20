"""One-shot: append a short amendment 8b to optimization/GPU_RUN_BOARD.md (BOM: `edit` cannot match)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "optimization" / "GPU_RUN_BOARD.md"

blk = """
Row 11 amendment 8b (2026-11-21 01:00, Moscow). **The third chain was restarted with one more point, so
that the machinery question is answered by data first.** `it36qatsmall3_chain` (relaunched 00:59:20,
lock re-taken) now runs, in order: `l1i8c` and `l1i8n` (the per-stream attribution pair on row 4's own
checkpoint), then `cn8` = full-matrix int8 on both streams with its own 124-step QAT decay. `cn8` is
8,704 card bits / 26,112 note bits, four times over the bars, so it is a DIAGNOSTIC and never a
candidate - but it has no rank truncation at all, so its tax separates the two candidate mechanisms
behind row 4's remaining immediate-head error: the rank-1 structure, or the 8-bit factor precision.
That decides whether spending the rank-1 headroom on a finer level (int16 factors, int16 shifts, or
unquantized factors) can help at all, which is the open decision recorded in amendment 8. `cn8` needs
its own QAT run rather than a re-eval of row 4's checkpoint, because evaluating a checkpoint with a
quantizer it was not trained for costs about +0.0012/+0.0015 and would swamp the diagnostic.
"""

with P.open("a", encoding="utf-8") as fh:
    fh.write(blk)

text = P.read_text(encoding="utf-8")
print(f"board lines now: {len(text.splitlines())}")
print(f"amendment 8b markers: {text.count('Row 11 amendment 8b')}")
