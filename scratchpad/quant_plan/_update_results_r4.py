"""RESULTS.md round 4: row 16 (card rank-3) is measured and passes."""
from pathlib import Path

P = Path("scratchpad/quant_plan/RESULTS.md")
with P.open(encoding="utf-8", newline="") as fh:
    lines = fh.read().split("\n")

hits = [i for i, l in enumerate(lines) if l.startswith("| 16 |")]
assert len(hits) == 1, hits
lines[hits[0]] = ("| 16 | **QAT card rank-3 int8 + note rank-1 int8 - the lowest tax of any row** | **2,048 (256)** | "
                  "**3,072 (384)** | *see note* | *see note* | **0.333808** | **0.306388** | **+0.000006** | "
                  "**+0.000176** | **PASS**: ub95 +0.000070/+0.000234; the ahead head is at zero and the "
                  "immediate head is 3x inside the ceiling |")

# add the rank ladder reading right after the speed paragraph
i0 = next(i for i, l in enumerate(lines) if l.startswith("**The rank-2 speed is fine on the CARD"))
i1 = next(i for i, l in enumerate(lines) if l.startswith("does)."))
assert i1 > i0
ladder = [
    "",
    "**The card's rank ladder (rows 15-16) confirms the mechanism and shows the knee.** Against row 0:",
    "",
    "| card | row | card bits | d ahead | d imm |",
    "|---|---|---:|---:|---:|",
    "| rank-1 int8 | 4 | 1,024 | +0.000503 | +0.001009 |",
    "| rank-2 int8 | 15 | 1,536 | +0.000046 | +0.000307 |",
    "| rank-3 int8 | 16 | 2,048 | +0.000006 | +0.000176 |",
    "",
    "So the immediate head is bought with rank and the curve flattens fast: the first extra rank removes",
    "70% of the tax, the second removes another 43% of what is left. Rank-3 is the axis-2 choice (lowest",
    "tax, +0.000006/+0.000176, so it leaves ~+0.0003 of the imm budget to spend on a smaller NOTE pool),",
    "while rank-2 is the axis-1 choice (1,536 card bits, 512 fewer). Rows 17-18 (int4 shifts) and the",
    "`it36notecut`/`it36rankprec`/`it36shrink` chains then search the (rank, precision, note-level) space",
    "for the smallest state that still passes.",
]
lines[i1 + 1:i1 + 1] = ladder

with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write("\n".join(lines))
print("round 4 done; lines:", len(lines))
