"""Round 5: Status line for row 16, the NOTES live-status bullet, and board amendment 12."""
from pathlib import Path

# --- RESULTS.md ---------------------------------------------------------------
P = Path("scratchpad/quant_plan/RESULTS.md")
t = P.read_text(encoding="utf-8")
old = "**Status: row 15 PASSES all six constraints - it is the first passing recipe.**"
new = ("**Status: rows 15 and 16 PASS all six constraints** (row 15 is the axis-1 choice at 1,536 card bits,\n"
       "row 16 the axis-2 choice at 2,048 with a near-zero tax).")
assert t.count(old) == 1
t = t.replace(old, new, 1)
P.write_text(t, encoding="utf-8")

# --- NOTES.md -----------------------------------------------------------------
P = Path("scratchpad/quant_plan/NOTES.md")
t = P.read_text(encoding="utf-8")
bullet = """* 2026-11-21 09:20 - **Row 16 (card rank-3 int8) is the lowest-tax row, and the rank ladder shows the
  knee.** `l3i8` (card rank-3 int8 + int8 shifts, note rank-1 int8; 2,048/3,072 bits) =
  0.333808/0.306388 = **+0.000006 ahead / +0.000176 imm** against row 0 (ub95 +0.000070/+0.000234). The
  immediate-head tax by rank: rank-1 +0.001009, rank-2 +0.000307 (row 15), rank-3 +0.000176 - the first
  extra rank removes 70%, the second another 43% of the rest. So rank-3 is the axis-2 choice (near-zero
  tax, and it leaves ~+0.0003 of the immediate-head budget to spend on a smaller NOTE pool) while rank-2
  is the axis-1 choice (512 fewer card bits). **Three new chains launched 09:15, queued behind the two
  running ones: `it36rankprec` (equal-bits rank-vs-precision probes: card rank-2 int4 at 768 bits vs row
  9's rank-1 int8, rank-4 int4 at 1,280 vs row 17's rank-2 int8, rank-7 int4 at 2,048 vs row 16's rank-3
  int8, and note rank-2 int4 at 2,304 vs row 17's note rank-1 int8), `it36longdecay` (rank-2 at 496 and
  992 steps against their own same-length twins - the absolute-loss axis, worth ~0.012-0.017 absolute),
  and `it36shrink` (four combos under row 15's 4,608 bits: 2,560 / 2,624 / 2,688 / 2,944).** Total queue
  ~13 points, roughly 8 h of GPU work.

"""
marker = "## Live status\n\n"
assert t.count(marker) == 1
t = t.replace(marker, marker + bullet, 1)
P.write_text(t, encoding="utf-8")

# --- board --------------------------------------------------------------------
P = Path("optimization/GPU_RUN_BOARD.md")
blk = """
Row 11 amendment 12 (2026-11-21 09:25, Moscow). **Row 16 (card rank-3 int8) is the lowest-tax row, and the
card rank ladder shows the knee.** `l3i8` = card rank-3 int8 + int8 shifts, note rank-1 int8
(2,048/3,072 bits) scores 0.333808/0.306388 = **+0.000006 ahead / +0.000176 imm** against row 0 (ub95
+0.000070/+0.000234). Immediate-head tax by card rank: rank-1 +0.001009, rank-2 +0.000307, rank-3
+0.000176 - so rank-3 is the axis-2 choice (near-zero tax, and it frees ~+0.0003 of the immediate-head
budget for a smaller note pool) and rank-2 (row 15) is the axis-1 choice with 512 fewer card bits.
Andrew asked for more gains on 2026-11-21 09:12 with the priority order 1) smaller state, 2) lower log
loss, 3) speed, and to queue lots of runs, so three new chains were launched at 09:15 behind the two
already running: `it36rankprec` (equal-bits rank-vs-precision probes - card rank-2 int4 at 768 bits vs
row 9's rank-1 int8, rank-4 int4 at 1,280 vs row 17's rank-2 int8, rank-7 int4 at 2,048 vs row 16's
rank-3 int8, note rank-2 int4 at 2,304 vs row 17's note rank-1 int8), `it36longdecay` (rank-2 at 496 and
992 steps against their own same-length fp32 twins, the absolute-loss axis), and `it36shrink` (four
combinations under row 15's 4,608 bits: 2,560 / 2,624 / 2,688 / 2,944). Eight new candidates were added
to `quant_sweep.CANDIDATES` (39 total). Queue depth ~13 points, roughly 8 h of GPU work. Andrew also
noted the deployed recipe is no longer his to change - another developer owns it and is running int4
cards + int8 notes - so this arm is an exploration for him, and the deployed 403/214-bit recipe is not
ours to touch.
"""
with P.open("ab") as fh:
    fh.write(blk.encode("utf-8"))
print("round 5 done; board lines:", P.read_text(encoding="utf-8-sig").count("\n") + 1)
