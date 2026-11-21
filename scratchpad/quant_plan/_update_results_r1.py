"""RESULTS.md update after the 2026-11-21 00:00-07:50 chains (rows 9-18, the cn8 mechanism verdict,
the per-stream attribution on row 4's checkpoint, the closed decay-length axis, and the gate framing).

Every replacement is anchored on exact existing text and asserted, so a re-wrap failure is loud.
"""
from pathlib import Path

P = Path("scratchpad/quant_plan/RESULTS.md")
with P.open(encoding="utf-8", newline="") as fh:
    text = fh.read()
nl = "\n" if "\r\n" not in text else "\r\n"

EDITS = []

# --- A: the gate framing on the table header ---------------------------------
EDITS.append((
    "`d` columns are paired per-user means against **row 0** (row 8 against its own twin, **row 0b**).",
    "`d` columns are paired per-user means against **row 0** - the identical fp32 decay with quantization\n"
    "OFF (row 8 against its own twin, row 0b; rows 11 and 15-18 against their own same-length twins). **That\n"
    "is the gate as Andrew defined it (2026-11-21): fp32 + no QAT + no quantization versus QAT + quantized.**\n"
    "A quantizer-trained checkpoint evaluated with the quantizer OFF (the \"twin\") is an ATTRIBUTION tool\n"
    "only; it is never the comparison number."
))

# --- B: the table rows 9-12 -> rows 9-18 -------------------------------------
old_rows_start = "| 9 | QAT rank-1 int8 factors + int4 shifts: SAME total bits as row 10, the bits moved to the factors |"
old_rows_end = "| 12 | QAT rank-1 int4, 992-step decay (only if 496 steps already beat 124 at these bits) | 512 (64) | 1,536 (192) | 2,586 | 0.903x | *queued* | *queued* | *queued* | *queued* | conditional in `it36qatsmall4`, vs `qnj_it36ctl992` |"
i0 = text.find(old_rows_start)
i1 = text.find(old_rows_end)
assert i0 > 0 and i1 > i0, (i0, i1)
i1 += len(old_rows_end)
NEW_ROWS = """| 9 | **QAT rank-1 int8 factors + int4 shifts: the bits belong in the FACTORS - BEST on axis 1** | **768 (96)** | **2,304 (288)** | 2,586 | 0.903x | **0.334318** | **0.307241** | **+0.000516** | **+0.001030** | **FAIL by a hair**, but 25% fewer bits than row 4 at the SAME tax (ub95 +0.000620/+0.001144, p 8.4e-31/8.9e-54) |
| 10 | QAT rank-1 int4 factors + int8 shifts: the same 768/2,304 bits moved to the SHIFTS | 768 (96) | 2,304 (288) | 2,586 | 0.903x | 0.334847 | 0.307873 | +0.001045 | +0.001661 | **FAIL**, 2.1x/3.3x over - so the factors need int8 and the shifts do not |
| 11 | QAT rank-1 int8 at a 992-step decay (8x the champion's 124 steps, relaxed rule) | 1,024 (128) | 3,072 (384) | 2,586 | 0.903x | 0.318088 | 0.291307 | +0.000874 | +0.001675 | **FAIL**, vs its own 992-step twin `qnj_it36ctl992` |
| 12 | QAT rank-1 int4 at a 992-step decay | 512 (64) | 1,536 (192) | 2,586 | 0.903x | *not run* | *not run* | - | - | chain 4's own guard: the 496-step `l1i4` tax (+0.001581) was worse than the 124-step one (+0.001050) |
| 13 | QAT rank-1 int4 at a 496-step decay (same length as row 8, half the bits) | 512 (64) | 1,536 (192) | 2,586 | 0.903x | 0.323240 | 0.296663 | +0.001581 | +0.002050 | **FAIL**, vs the 496-step twin (row 0b) |
| 14 | QAT rank-1 int4 factors + int2 shifts at a 496-step decay | 384 (48) | 1,152 (144) | 2,586 | 0.903x | 0.323603 | 0.297312 | +0.001944 | +0.002698 | **FAIL**, vs the 496-step twin, 3.9x/5.4x over |
| 15 | QAT card **rank-2** int8 + note rank-1 int8: the card's rank doubled inside its bar | 1,536 (192) | 3,072 (384) | *see note* | *see note* | *running* | *running* | *running* | *running* | `it36rank`, vs row 0 |
| 16 | QAT card **rank-3** int8 + note rank-1 int8 | 2,048 (256) | 3,072 (384) | *see note* | *see note* | *running* | *running* | *running* | *running* | `it36rank`, vs row 0 |
| 17 | QAT card rank-2 int8 + int4 shifts, note rank-1 int8 + int4 shifts | 1,280 (160) | 2,304 (288) | *see note* | *see note* | *running* | *running* | *running* | *running* | `it36rank`, vs row 0 |
| 18 | QAT card rank-3 int8 + int4 shifts, note rank-1 int8 + int4 shifts | 1,792 (224) | 2,304 (288) | *see note* | *see note* | *running* | *running* | *running* | *running* | `it36rank`, vs row 0 |"""
text = text[:i0] + NEW_ROWS + text[i1:]

# --- C: the speed note under the table ---------------------------------------
EDITS.append((
    "Rows 5-7, 9 and 10 all use the rank-1 path, whose engine speed was measured once (`l1i4` and `l1i8` both\n"
    "2,586 rev/s = 0.903x); the factor/shift bit widths do not change that path, so their speed cells are\n"
    "that measurement rather than a per-candidate run.",
    "Rows 5-7, 9-11 and 13-14 all use the rank-1 path, whose engine speed was measured once (`l1i4` and\n"
    "`l1i8` both 2,586 rev/s = 0.903x); the factor/shift bit widths do not change that path, so their speed\n"
    "cells are that measurement rather than a per-candidate run.\n"
    "\n"
    "**Rows 15-18 have no deploy speed yet, and the deploy engine is the problem.** They change the CARD's\n"
    "rank, and the engine reconstructs the full 32x32 matrix every step for rank >= 2 instead of using a\n"
    "fused readout, so the engine's rank-2 measurement is 0.629x rev/s / 0.448x states/s - below the 0.75x\n"
    "floor. Their loss numbers are still the right first measurement (the torch eval path mirrors the\n"
    "engine's rounding), but they cannot be deployed until the engine gets a fused rank-r readout."
))

# --- D: the bits-vs-tax curve reading, now with the factor/shift answer ------
EDITS.append((
    "trade-off curve rather than a hard limit - and the *token shifts* look like the expensive side of the\n"
    "rank-1 split (int4 -> int2 shifts alone cost +0.000395/+0.000598 at row 6). Rows 9 and 10 test that\n"
    "directly: they hold the total at 768/2,304 bits and move the bits between the factors and the shifts.",
    "trade-off curve rather than a hard limit.\n"
    "\n"
    "**Rows 9-10 answer the bit-split question, and the answer is counter-intuitive: the bits belong in the\n"
    "FACTORS, not the shifts.** Both rows spend 768/2,304 bits. Row 9 gives the factors int8 and the shifts\n"
    "int4 (+0.000516/+0.001030) and lands within +0.00002 of row 4, which spends 1,024/3,072 - i.e. the\n"
    "shift vectors can drop from int8 to int4 for FREE, saving 25% of the bits. Row 10 does the opposite\n"
    "(int4 factors, int8 shifts) and is as bad as row 5 (+0.001045/+0.001661), even though it has 50% MORE\n"
    "bits. The factors are rank-1's real content; the shifts ride along."
))

# --- E: the decay-length axis, closed ---------------------------------------
old_len_start = "**Row 0b does NOT rescue the tax - the 496-step hypothesis is refuted.**"
old_len_end = "If the tax keeps rising at 992, this route is closed for good and only bit allocation is left."
j0 = text.find(old_len_start)
j1 = text.find(old_len_end)
assert j0 > 0 and j1 > j0, (j0, j1)
j1 += len(old_len_end)
NEW_LEN = """**The decay-length axis is CLOSED (2026-11-21 06:12), and it is closed in the awkward direction.**
Three lengths of the same `l1i8` recipe now have their own same-length fp32 twin:

| decay | quantized | fp32 twin | d ahead | d imm |
|---|---|---|---|---|
| 124 (row 4) | 0.334305 / 0.307221 | 0.333802 / 0.306211 | +0.000503 | +0.001009 |
| 496 (row 8) | 0.322791 / 0.296287 | 0.321659 / 0.294614 | +0.001132 | +0.001674 |
| 992 (row 11) | 0.318088 / 0.291307 | 0.317213 / 0.289632 | +0.000874 | +0.001675 |

The tax does not fall at any length, and it roughly doubles past 124 steps on the immediate head. Andrew
relaxed the step budget (2026-11-21 01:00: "instead of using 124 steps, you can use more steps ... if it
ultimately ends up turning a failing recipe into a winning one, it's fine"), and this is the answer: the
longer decays do not turn a failing recipe into a winning one. What they DO give is a much better model
in ABSOLUTE terms - the 992-step quantized run is 0.0162/0.0159 BETTER than row 4 on both heads, because
the fp32 twin itself improves by 0.0166/0.0170 between 124 and 992 steps. So for the rest of this arm the
length is held at the champion's 124 steps (comparable, and the cheapest), and `l1i4` at 992 was correctly
skipped by chain 4's own guard (the 496-step `l1i4` tax was already worse than the 124-step one)."""
text = text[:j0] + NEW_LEN + text[j1:]

# --- F: Status paragraph -----------------------------------------------------
EDITS.append((
    "**Status: nothing passes the +0.0005 bar yet.** Row 4 is the best point: ahead is essentially at the\n"
    "ceiling, the immediate head is twice over. Rows 3 and 1 are 2-3x over on both heads, and row 2 shows\n"
    "that QAT does not rescue a naive full-matrix quantizer.",
    "**Status: nothing passes the +0.0005 bar yet, but the frontier moved twice.** Row 9 is now the best\n"
    "axis-1 point (768/2,304 bits - 25% smaller than row 4 - at the same tax), and the mechanism is settled\n"
    "(see \"the tax is RANK TRUNCATION\" below): precision is free at 8 bits, so the remaining +0.001009 on\n"
    "the immediate head must be bought with RANK, and the card stream is the one that needs it."
))

for old, new in EDITS:
    assert text.count(old) == 1, (text.count(old), old[:70])
    text = text.replace(old, new, 1)

with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("rows/readings updated; lines:", text.count("\n") + 1)
