"""One-shot (2026-11-21 23:05, Moscow): the NOTES.md Live-status bullet for row 23 + the corruption
column, and the RECORDED.txt stamp for it36l2i4s4. Refuses to run twice."""
import pathlib

P = pathlib.Path(__file__).resolve().parents[1] / "NOTES.md"

BULLET = """* 2026-11-21 23:05 - **Row 23 `l2i4s4` FAILS, and RESULTS.md now carries Andrew's `corruption` column.** `l2i4s4` holds the card at rank 2 with **int4 factors** (the cheap precision) + int4 shifts and the note at row 9's rank-1 int8 + int4 shifts: 768 card / 2,304 note bits - the SAME bit budget as row 9, where the card is rank-1 int8. It scored 0.334068/0.306792 = **+0.000266 ahead / +0.000580 imm** against `qnj_it36ctl` (ub95 +0.000337/+0.000661, p 6.9e-16/1.5e-38): ahead is 0.53x of the ceiling, **imm is 1.16x over it**. The informative part is the head-to-head at identical bits: rank-2 int4 beats rank-1 int8 by about **+0.00025 ahead / +0.00045 imm**, so at a fixed card budget extra RANK buys more than extra factor precision - but it does not rescue the imm head, which still needs the int8-quality 2,304-bit note. Recorded as RESULTS.md row 23 and stamped in `autorun/RECORDED.txt`. **New column**: `corruption`, requested by Andrew at 22:3x - `pending` for row 0 (the fp32 control) and the four passing rows 15-18, which the queued `it36corrupt` chain will score, and `-` for every row that fails the log-loss gate (not a deploy candidate, so it is never checked). The column legend lives in RESULTS.md, and `autorun/QUEUE.md` section 4 now tells the hourly supervisor that the main table has 12 columns and to write `pending` rather than a guess.
"""

with P.open(encoding="utf-8", newline="") as fh:
    text = fh.read()
if "Row 23 `l2i4s4` FAILS" in text:
    raise SystemExit("bullet already present")
for nl in ("\n", "\r\n"):
    anchor = f"## Live status{nl}{nl}"
    if anchor in text:
        text = text.replace(anchor, anchor + BULLET.replace("\n", nl), 1)
        break
else:
    raise SystemExit("Live status anchor not found")
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("bullet inserted into", P)

R = pathlib.Path(__file__).resolve().parents[1] / "autorun" / "RECORDED.txt"
with R.open(encoding="utf-8", newline="") as fh:
    rt = fh.read()
if "it36l2i4s4" in rt:
    raise SystemExit("stamp already present")
if not rt.endswith("\n"):
    rt += "\n"
rt += "it36l2i4s4\n"
with R.open("w", encoding="utf-8", newline="") as fh:
    fh.write(rt)
print("stamp appended to", R)
