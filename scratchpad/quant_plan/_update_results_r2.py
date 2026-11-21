"""RESULTS.md round 2: the mechanism section, the row-4 attribution table, and the iteration log."""
from pathlib import Path

P = Path("scratchpad/quant_plan/RESULTS.md")
with P.open(encoding="utf-8", newline="") as fh:
    text = fh.read()

MECH = """**The tax is RANK TRUNCATION, not precision (settled 2026-11-21 04:33 by the `cn8` diagnostic).**

`cn8` is full-matrix int8 on both streams - no rank truncation at all - and it is 4x over both bars
(8,704 / 26,112 bits), so it is a DIAGNOSTIC, never a candidate. Its tax against row 0 is
**+0.000033 / +0.000022**: an 8-bit state is essentially lossless when it is not truncated. The rank-1
family's +0.000503/+0.001009 is therefore the rank-1 TRUNCATION, not the 8-bit factor precision, and no
finer level can buy it back. The fix has to add RANK.

Which stream should get the rank? The per-stream attribution pair ran on row 4's OWN checkpoint (its own
quantizer, so there is no quantizer mismatch), each quantizing one stream and leaving the other in fp32:

| run | card bits | note bits | ahead | imm | d ahead | d imm |
|---|---:|---:|---:|---:|---:|---:|
| row 4 `l1i8` (both streams rank-1 int8) | 1,024 | 3,072 | 0.334305 | 0.307221 | +0.000503 | +0.001009 |
| `l1i8c`, CARD only | 1,024 | fp32 | 0.335084 | 0.307353 | +0.001282 | +0.001142 |
| `l1i8n`, NOTE only | fp32 | 3,072 | 0.334175 | 0.306470 | +0.000373 | +0.000258 |

The note stream at rank-1 int8 costs +0.000373/+0.000258 on its own - INSIDE the ceiling on both heads.
The card stream carries the immediate-head tax by itself (+0.001142, which is even more than the two
streams cost together, because each single-stream run has the other stream mismatched against its QAT
training). So the rank must go to the CARD, and the card's 2,176-bit bar has room: 1,024 bits at rank-1
int8 + int8 shifts, 1,536 at rank-2, 2,048 at rank-3 (128 bits under the bar), 2,176 exactly at rank-4
int8 + int2 shifts. Rows 15-18 measure that (running, 2026-11-21 07:45).

Second consequence: the note pool (3,072 of row 4's 4,096 bits) is where the BITS are while the card is
where the TAX is, so a note cut is worth measuring as well - rank-1 int4 (1,536 bits) or rank-1 int4
factors + int2 shifts (1,152). `it36notecut` runs that (`nr4`, `nr4s2`) and then the combined
`l2i8s4n4` / `l3i8s4n4` points, gated on the rank ladder actually helping.

"""

anchor = "**The decay-length axis is CLOSED (2026-11-21 06:12)"
assert text.count(anchor) == 1
text = text.replace(anchor, MECH + anchor, 1)

# --- the attribution table: add the row-4-checkpoint pair ---------------------
old = ("Reading: the two streams are close to additive (0.000943/0.001397 when both are quantized). Card\n"
       "hurts the immediate head most; note hurts the ahead head slightly more.")
new = ("Reading: the two streams are close to additive (0.000943/0.001397 when both are quantized). Card\n"
       "hurts the immediate head most; note hurts the ahead head slightly more.\n"
       "\n"
       "**The same attribution on row 4's own checkpoint, QAT-trained for BOTH streams (2026-11-21 03:32),\n"
       "i.e. with the row 0 gate and no quantizer mismatch on the quantized stream:**\n"
       "\n"
       "| Run | Method | card bits (B) | note bits (B) | ahead LogLoss | imm LogLoss | d ahead | d imm |\n"
       "|---|---|---:|---:|---:|---:|---:|---:|\n"
       "| d3 | `l1i8c`: rank-1 int8 on the CARD stream only, note fp32 | 1,024 (128) | 104,448 (13,056) | 0.335084 | 0.307353 | +0.001282 | +0.001142 |\n"
       "| d4 | `l1i8n`: rank-1 int8 on the NOTE stream only, card fp32 | 34,816 (4,352) | 3,072 (384) | 0.334175 | 0.306470 | +0.000373 | +0.000258 |\n"
       "\n"
       "This pair is the one that decides where the rank goes (see \"the tax is RANK TRUNCATION\" above):\n"
       "the note is inside the ceiling on both heads by itself and the card owns the immediate-head tax.\n"
       "\n"
       "**Mechanism diagnostic (2026-11-21 04:33):** `cn8`, full-matrix int8 on both streams with NO rank\n"
       "truncation, 8,704/26,112 bits, 4x over both bars - DIAGNOSTIC ONLY, never a candidate. It scores\n"
       "0.333834/0.306234 = **+0.000033/+0.000022** against row 0, so 8-bit precision of the FULL state is\n"
       "free and the rank-1 tax is truncation.")
assert text.count(old) == 1
text = text.replace(old, new, 1)

# --- the iteration log --------------------------------------------------------
old = ("- **Rows 9-10, `it36qatsmall2` (queued, took the GPU at 00:51:57)**: `l1i4` at 496 steps against\n"
       "  `qnj_it36ctl496` (the same decay-length question at the cheapest allocation), then the\n"
       "  768/2,304-bit attribution pair `l1i8s4` (int8 factors + int4 shifts) and `l1i4s8` (int4 factors +\n"
       "  int8 shifts), then `l1i4s2` at 496 steps. Its own guard skips `l1i2`@496, because the 124-step\n"
       "  `l1i2` point is +0.003604 and the threshold is +0.0025.")
new = ("- **Rows 9-10 plus the 496-step ladder, `it36qatsmall2` (finished 2026-11-21 03:08:54, rc=0)**:\n"
       "  `l1i4` at 496 steps = +0.001581/+0.002050 against `qnj_it36ctl496` (worse than `l1i8`@496, as\n"
       "  expected at half the bits); the 768/2,304-bit split pair `l1i8s4` = +0.000516/+0.001030 (row 9,\n"
       "  the new axis-1 best - the int8 factors keep the tax while the int4 shifts save 25% of the bits)\n"
       "  and `l1i4s8` = +0.001045/+0.001661 (row 10); then `l1i4s2` at 496 steps = +0.001944/+0.002698.\n"
       "  Its own guard correctly skipped `l1i2`@496.\n"
       "- **`it36qatsmall3` (finished 2026-11-21 04:33:14, rc=0)**: the per-stream attribution pair on row\n"
       "  4's own checkpoint (`l1i8c` +0.001282/+0.001142, `l1i8n` +0.000373/+0.000258) and the `cn8`\n"
       "  mechanism diagnostic with its own 124-step QAT decay (+0.000033/+0.000022 - the decisive result).\n"
       "- **Rows 11-12, `it36qatsmall4` (finished 2026-11-21 06:12:50, rc=0)**: the 992-step fp32 twin\n"
       "  `qnj_it36ctl992` = 0.317213/0.289632, i.e. 0.016589/0.016579 better than row 0 (that gain is the\n"
       "  extra training alone) and 0.025600/0.024640 better than the iter36 champion `qnj_it36_124`; then\n"
       "  `l1i8` at 992 = +0.000874/+0.001675 against that twin, and `l1i4`@992 skipped by the guard.\n"
       "- **Rows 15-18, `it36rank` (launched 2026-11-21 07:45, RUNNING)**: the card rank ladder - `l2i8`\n"
       "  (card rank-2 int8, 1,536/3,072), `l3i8` (card rank-3 int8, 2,048/3,072), `l2i8s4` (1,280/2,304)\n"
       "  and `l3i8s4` (1,792/2,304). This is the direct test of the `cn8` verdict: if the rank-1 tax is\n"
       "  truncation, buying rank inside the card's bar is what removes it.\n"
       "- **`it36notecut` (queued behind `it36rank` at 2026-11-21 07:48)**: the note pool cut - `nr4` (note\n"
       "  rank-1 int4 factors + int4 shifts, 1,024/1,536), `nr4s2` (1,024/1,152), then, only if the best\n"
       "  card-rank imm delta is under +0.00070, the combined `l2i8s4n4` (1,280/1,536) and `l3i8s4n4`\n"
       "  (1,792/1,536).")
assert text.count(old) == 1
text = text.replace(old, new, 1)

with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("mechanism section + attribution + log done; lines:", text.count("\n") + 1)
