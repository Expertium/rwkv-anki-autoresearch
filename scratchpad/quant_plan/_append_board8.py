"""One-shot: append Row 11 amendment 8 to optimization/GPU_RUN_BOARD.md (BOM: `edit` cannot match)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "optimization" / "GPU_RUN_BOARD.md"

blk = """
Row 11 amendment 8 (2026-11-21 01:00, Moscow). **The bit ladder is closed, and the longer-decay route to
compliance is REFUTED.** Rows 7 and 8 of `RESULTS.md` are scored against the identical unquantized runs:

  * row 7, `l1i2` (rank-1 int2 factors + int2 shifts, 256/768 bits) = 0.337406/0.311143 =
    +0.003604 ahead / +0.004932 imm vs the 124-step control = 7.2x/9.9x over the ceiling.
  * row 8, `l1i8` at 496 steps (row 4's exact bits) = 0.322791/0.296287 = **+0.001132/+0.001674** vs its
    own 496-step fp32 twin (row 0b). The tax MORE than doubles versus the 124-step pair
    (+0.000503/+0.001009) even though the absolute quantized model is 0.0115 better than row 4: the fp32
    twin keeps improving faster than a fixed-bit quantizer can follow. Compliance must therefore be won
    at the champion's own decay length, and more training is not a lever.

The axis-1 curve is now complete and convex (card/note bits -> d ahead / d imm vs the 124-step control):
1,024/3,072 -> +0.000503/+0.001009; 512/1,536 -> +0.001050/+0.001682; 384/1,152 -> +0.001445/+0.002280;
256/768 -> +0.003604/+0.004932. Row 4 remains the best point on both heads and is 2.1x/4.3x under the
constraint-2 bars, so the bars are not the binding constraint - the +0.0005 loss ceiling is, and the
immediate head is the blocker.

Run state: `it36qatsmall_chain` ENDED at 00:51:55 (all four points scored, rows 5-8).
`it36qatsmall2_chain` took the GPU at 00:51:57 and is running `l1i4` at 496 steps (the same longer-decay
question at the cheapest allocation); it will then run `l1i8s4` and `l1i4s8` (the 768/2,304-bit
attribution pair) and `l1i4s2` at 496 steps, and its own guard skips `l1i2`@496 because the 124-step
`l1i2` point is +0.003604 (over its +0.0025 threshold). `it36qatsmall3_chain` is queued behind it
(lock-guarded, detached 00:55): the per-stream attribution pair `l1i8c` (card rank-1 int8 only) and
`l1i8n` (note rank-1 int8 only) on row 4's own checkpoint, which splits row 4's ahead/imm tax into a
card part and a note part. Both are attribution rows only - they leave one stream in fp32, so they do
not meet constraint 2 and are never candidates.

OPEN DECISION FOR ANDREW (no run is waiting on it): row 4's allocation spends 1,024 of the allowed
2,176 bits per layer, and inside the rank-1 fast path the only ways to spend the headroom are finer
levels - int8 factors + int16 shifts (1,536 bits/layer), int16 factors + int8 shifts (2,048), or
unquantized factors + int2 shifts (2,176, exactly at the bar). Each needs a small machinery change in
BOTH the QAT path (`rwkv/architecture.py` `_QMAX`) and the Rust engine's level handling plus the bit
accounting, so it is a research-direction choice rather than a screening point. The alternative is to
accept row 4 as the best available recipe (ahead at the ceiling, imm 2x over) or to change the state
STRUCTURE (per-row scales, a diagonal term) rather than the level set.

The deployed 403 card bits / 214 note bits stay untouched; no deploy-contract change is proposed.
"""

with P.open("a", encoding="utf-8") as fh:
    fh.write(blk)

text = P.read_text(encoding="utf-8")
print(f"board lines now: {len(text.splitlines())}")
print(f"amendment 8 markers: {text.count('Row 11 amendment 8')}")
