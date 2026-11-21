"""RESULTS.md round 3 (line-based): row 15 is a PASSING recipe and the rank-2 speed is measured."""
from pathlib import Path

P = Path("scratchpad/quant_plan/RESULTS.md")
with P.open(encoding="utf-8", newline="") as fh:
    lines = fh.read().split("\n")

# --- 1. row 15 -----------------------------------------------------------------
hits = [i for i, l in enumerate(lines) if l.startswith("| 15 |")]
assert len(hits) == 1, hits
lines[hits[0]] = ("| 15 | **QAT card rank-2 int8 + note rank-1 int8 - THE FIRST PASSING RECIPE** | **1,536 (192)** | "
                  "**3,072 (384)** | **2,324** | **0.812x** | **0.333848** | **0.306518** | **+0.000046** | "
                  "**+0.000307** | **PASS**: ub95 +0.000112/+0.000369, both heads inside the ceiling, and the "
                  "deploy engine runs it at 0.812x (>= 0.75x) |")

# --- 2. the speed paragraph ----------------------------------------------------
i0 = next(i for i, l in enumerate(lines) if l.startswith("**Rows 15-18 have no deploy speed yet"))
i1 = next(i for i, l in enumerate(lines) if l.startswith("engine's rounding), but they cannot be deployed"))
assert i1 > i0, (i0, i1)
new_speed = [
    "**The rank-2 speed is fine on the CARD; the earlier 0.629x was a both-streams measurement.**",
    "`l2i4` (rank-2 on card AND note, all four note layers) is 0.629x rev/s / 0.448x states/s, which is what",
    "made rank >= 2 look undeployable. Row 15 puts rank 2 on the CARD only - 1 layer, and the cheap stream -",
    "and the deploy engine measures it at **2,323.6 rev/s = 0.812x** (states/s 0.660x, median per-user 0.803x),",
    "i.e. ABOVE the 0.75x floor. It was measured while a QAT decay was training, so it is about 3% pessimistic",
    "(`l1i8` reads 0.875x in that run against 0.903x on the idle machine); an idle-machine re-measurement is",
    "still owed. Row 17 (`l2i8s4`) 0.800x and row 18 (`l3i8s4`) 0.800x were measured in the same run. No engine",
    "change is needed: the deploy engine already parses `card:2:int8`, and the torch eval path mirrors its",
    "rounding (the 3-user deploy grid ranks rank-2 below rank-1 on the immediate head, as the 750-user eval",
    "does).",
]
lines[i0:i1 + 1] = new_speed

# --- 3. Status -----------------------------------------------------------------
j0 = next(i for i, l in enumerate(lines) if l.startswith("**Status: nothing passes the +0.0005 bar yet"))
j1 = next(i for i, l in enumerate(lines) if l.startswith("the immediate head must be bought with RANK"))
assert j1 > j0, (j0, j1)
new_status = [
    "**Status: row 15 PASSES all six constraints - it is the first passing recipe.** Card 1,536 bits (2.83x",
    "smaller than naive int4, bar 2,176) and note 3,072 bits (2.13x smaller, bar 6,528); both heads inside the",
    "+0.0005 ceiling with their upper bounds too (+0.000046 ub95 +0.000112 ahead, +0.000307 ub95 +0.000369",
    "imm); the 750-user roster; and the deploy engine at 0.812x rev/s (>= 0.75x). It buys the immediate head",
    "with RANK on the card, exactly what the `cn8` verdict predicted (see \"the tax is RANK TRUNCATION\"",
    "below). Row 9 stays the best axis-1 rank-1 point (768/2,304 bits at row 4's tax) for reference.",
]
lines[j0:j1 + 1] = new_status

with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write("\n".join(lines))
print("round 3 done; lines:", len(lines))
