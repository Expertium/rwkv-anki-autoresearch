"""Append board amendment 10 (the cn8/attribution/row-9/length-axis results + the rank chains)."""
from pathlib import Path

P = Path("optimization/GPU_RUN_BOARD.md")
blk = """
Row 11 amendment 10 (2026-11-21 08:00, Moscow). **Four chains finished overnight and the mechanism is
settled: the rank-1 tax is RANK TRUNCATION, not precision - and row 9 is a 25%-smaller state at the same
tax.** `it36qatsmall2` ended 03:08:54, `it36qatsmall3` 04:33:14, `it36qatsmall4` 06:12:50, all rc=0.
Key results: (1) `cn8`, the full-matrix int8 diagnostic with NO rank truncation (8,704/26,112 bits, 4x
over the bars, never a candidate) scores +0.000033/+0.000022 against row 0, so an 8-bit state is
lossless when it is not truncated and a finer level cannot help - the fix must add RANK. (2) The
per-stream attribution on row 4's own checkpoint splits it: card-only +0.001282/+0.001142, note-only
+0.000373/+0.000258, so the note is inside the ceiling by itself and the card owns the immediate head.
(3) Row 9 `l1i8s4` (int8 factors + int4 shifts, 768/2,304 bits) = +0.000516/+0.001030, i.e. row 4's tax
for 25% fewer bits: the shifts can drop int8 -> int4 for free and the bits belong in the factors. (4) The
decay-length axis is CLOSED - `l1i8`'s tax is +0.000503/+0.001009 at 124, +0.001132/+0.001674 at 496 and
+0.000874/+0.001675 at 992 - although the 992-step quantized run is 0.0162/0.0159 BETTER than row 4 in
absolute terms (its fp32 twin is 0.317213/0.289632), so absolute and relative readings disagree. Also:
`it36qatsmall2`'s own guard correctly skipped `l1i2`@496 and chain 4's skipped `l1i4`@992. RUNNING:
`it36rank` (launched 07:45) measures the card rank ladder - rows 15-18, card rank-2/3 int8 with int8 or
int4 shifts, 1,280-2,048 card bits, note unchanged. QUEUED behind it: `it36notecut` (note rank-1 int4
pool cut, plus the gated `l*i*s4n4` combinations only if the best card-rank imm delta comes in under
+0.00070). BLOCKER: rank >= 2 has no fused path in the deploy engine (measured 0.629x rev/s, 0.448x
states/s), so any rank >= 2 recipe needs a fused rank-r readout in the engine before it can be deployed
- that is a deploy-path change and needs Andrew's approval. GPU at amendment time: `it36rank`'s `l2i8`
QAT decay, chain `it36notecut` waiting.
"""
with P.open("ab") as fh:
    fh.write(blk.encode("utf-8"))
text = P.read_text(encoding="utf-8-sig")
print("amendment 10 markers:", text.count("Row 11 amendment 10"))
print("board lines:", text.count("\n") + 1)
