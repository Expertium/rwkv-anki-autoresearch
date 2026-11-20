"""One-shot: append Row 11 amendment 7 to optimization/GPU_RUN_BOARD.md (BOM: `edit` cannot match)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "optimization" / "GPU_RUN_BOARD.md"

blk = """
Row 11 amendment 7 (2026-11-21 00:00, Moscow). **The axis-1 bit ladder is running and the curve is
steep.** Two more points are scored against the identical 124-step fp32 control (`qnj_it36ctl`,
0.333802/0.306211):

  * row 5, `l1i4` (rank-1 int4 factors + int4 shifts, card 512 bits / note 1,536 bits) =
    0.334852/0.307894 = **+0.001050 ahead / +0.001682 imm** (ub95 +0.001164/+0.001841, p 2e-63/3.1e-75)
    - 2.1x/3.4x over the +0.0005 ceiling, and roughly double row 4's tax for half the bits.
  * row 6, `l1i4s2` (int4 factors + int2 shifts, 384/1,152 bits) = 0.335247/0.308491 =
    **+0.001445 ahead / +0.002280 imm** (ub95 +0.001592/+0.002464, p 3.0e-66/1.9e-88) - 2.9x/4.6x over.

So the token-shift vectors look like the expensive side of the rank-1 split (int4 -> int2 shifts cost
+0.000395/+0.000598 on their own). Both streams are still far under their constraint-2 bars (card 512
or 384 vs 2,176; note 1,536 or 1,152 vs 6,528), so axis 1 is a trade-off curve, not a hard limit.

Still running, one GPU job at a time: `it36qatsmall_chain` finished the `l1i2` smoke, started the
`l1i2` 124-step QAT at 23:44:46, and will then run `l1i8` at 496 steps against the row 0b twin
(`qnj_it36ctl496`). Its last two points are ROWS 7 and 8 of `RESULTS.md`.

QUEUED, second screening chain: `scratchpad/quant_plan/it36qatsmall2_chain.py`, launched detached
2026-11-20 23:47. It waits for `it36qatsmall_chain` to log its end marker with zero matching processes
(it does not count itself, and it takes `scratchpad/quant_plan/it36qatsmall2_chain.lock` so a double
launch cannot happen; delete that lock by hand after an outage). Its points, in order:
  * `l1i4` at 496 steps (tag `it36l1i4496`) vs `qnj_it36ctl496` - can the longer decay absorb row 5's
    +0.001050/+0.001682 at 512/1,536 bits?
  * `l1i8s4` and `l1i4s8` at 124 steps (tags `it36small8s4`, `it36small4s8`) - the new attribution pair
    in `quant_sweep.CANDIDATES`: both are 768 card / 2,304 note bits, with the bits moved between the
    rank-1 factors and the shifts, so the pair separates "factor error" from "shift error".
  * `l1i4s2` at 496 steps, then `l1i2` at 496 steps (the last one only if the 124-step `l1i2` deltas
    are within +0.0025, because a longer decay will not rescue a catastrophic allocation).
Nothing here changes the deploy contract: the deployed 403 card bits / 214 note bits stay untouched.
"""

with P.open("a", encoding="utf-8") as fh:
    fh.write(blk)

text = P.read_text(encoding="utf-8")
print(f"board lines now: {len(text.splitlines())}")
print(f"amendment 7 markers: {text.count('Row 11 amendment 7')}")
