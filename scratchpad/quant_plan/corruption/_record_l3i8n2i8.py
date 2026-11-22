"""Record chain 14 point 2 (`l3i8n2i8`, RESULTS row 30) in RESULTS.md, NOTES.md and RECORDED.txt.

Both files mix CRLF and LF, so every anchor is matched as RAW BYTES (no newline translation) and each
one is asserted to be unique before it is replaced. Nothing is invented: every number written here
comes from `quant_sweep_it36l3i8n2i8.json` (the landed sweep) and `rust_speed_fastrank_{on,off}.json`
(the measured deploy-engine speed for this config).

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/_record_l3i8n2i8.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "scratchpad" / "quant_plan" / "RESULTS.md"
NOTES = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"
RECORDED = ROOT / "scratchpad" / "quant_plan" / "autorun" / "RECORDED.txt"

ROW30 = (
    "| 30 | QAT card **rank-3 int8** + int4 shifts (row 18's card), note **rank-2 int8** + int4 "
    "shifts - the note's own rank ladder, second step (`l3i8n2i8`) | 1,792 (224) | 3,840 (480) | "
    "2,393 | 0.791x | 0.333658 | 0.306198 | -0.000144 | -0.000014 | **PASS on BOTH heads, and the "
    "lowest-tax row measured so far**: ahead is BETTER than the anchor by 0.000144 (ub95 -0.000121, "
    "p_worse 1.0000) and the immediate head's point estimate is 0.000014 BELOW the anchor (ub95 "
    "+0.000002 - the 95% bound sits on the anchor, p_worse 0.851, i.e. no evidence of worsening), so "
    "this is the first row whose imm head is not measurably worse than the unquantized-equivalent "
    "run at all; it spends 448 more card bits than row 29 (1,792 vs 1,280) for 0.000029/0.000117 "
    "better on the two heads, and its rank-2 NOTE path clears the speed floor at 0.791x with the "
    "fast rank-r solver (`rust_speed_fastrank_on.json`; 0.611x without it) | pending |"
)

# The stale sentence: it still says the rank>=2 note cannot be deployed, which E2 settled.
OLD_TAIL = (
    b"The blocker has moved to the ENGINE: a rank-2 note path reads 0.600x (0.75x floor), so row 29 "
    b"is not deployable until a fast rank-r note path exists (corruption: pending, `it36corrupt2`)."
)
NEW_TAIL = (
    b"The blocker had moved to the ENGINE: a rank-2 note path reads 0.600x on the reference solver "
    b"(0.75x floor), so the engine needed a FAST rank-r path, not a parser change - `RWKV_LOWRANK_FAST=1` "
    b"(E2, 2026-11-22) supplies it and measures 0.803x for this row (see the fast-solver section above), "
    b"so row 29 is a deploy candidate once its corruption check passes (corruption: pending, "
    b"`it36corrupt2`)."
)

PARA = (
    "**Row 30 (`l3i8n2i8`) lands and takes over as the lowest-tax row: the note's rank ladder pays for "
    "itself in loss at 5,632 bits.** With row 18's card held (rank-3 int8, 1,792 bits), the note goes to "
    "rank-2 int8 + int4 shifts (3,840 bits) and scores 0.333658/0.306198 = **-0.000144 ahead / -0.000014 "
    "imm** (ub95 -0.000121/+0.000002, p_worse 1.0000/0.851) against `qnj_it36ctl`. Both point estimates "
    "are at or below the anchor and the immediate head's 95% bound sits exactly on it, so this is the "
    "first row in the ladder whose immediate head carries no measurable tax at all. Against row 29 (same "
    "note width, card rank 2 at 1,280 bits) it is 0.000029 ahead / 0.000117 imm BETTER for 448 more card "
    "bits, so the card's rank-3 step buys real immediate-head loss. **But rank 2 does not make the note "
    "SMALLER - it makes it less taxed**: at 3,840 note bits it costs 1,536 more bits than the int8 "
    "rank-1 note of rows 17-18 (2,304) that already passes, and the equal-bit rank-vs-precision swap on "
    "the note was already settled the other way by row 26 (`n2i4s4`, rank-2 int4 at row 17's exact "
    "2,304 note bits) which failed by a hair. So on axis 1 (smallest state) row 29 stays the preference "
    "at 5,120 bits and this row is a LOSS instrument: it removes the note's tax, which is the "
    "prerequisite for shrinking the note later. Speed 0.791x with the fast rank-r solver (0.611x "
    "without, `rust_speed_fastrank_{on,off}.json`), above the 0.75x floor; corruption pending "
    "(`it36corrupt2` already carries this tag)."
)

BULLET = (
    "* 2026-11-22 14:05 - **Chain 14 point 2 lands as RESULTS row 30 (`l3i8n2i8`, card rank-3 int8 + "
    "note rank-2 int8 + int4 shifts, 1,792/3,840 = 5,632 bits): PASS on BOTH heads, and the lowest tax "
    "measured so far** - 0.333658/0.306198 = **-0.000144 ahead / -0.000014 imm** vs `qnj_it36ctl` (ub95 "
    "-0.000121/+0.000002, p_worse 1.0000/0.851). The immediate head's 95% bound sits exactly ON the "
    "anchor, so for the first time the imm head carries no measurable tax at all. It beats row 29 by "
    "0.000029/0.000117 for 448 more card bits, but rank 2 makes the note LESS TAXED, not SMALLER (3,840 "
    "bits vs the 2,304-bit int8 rank-1 note of rows 17-18 that already passes), so row 29 remains the "
    "axis-1 preference at 5,120 bits and this row is the loss instrument. Speed 0.791x with "
    "`RWKV_LOWRANK_FAST=1` (0.611x without). Corruption: pending (`it36corrupt2`). Also repaired the "
    "stale RESULTS.md sentence that still said a rank>=2 note is not deployable. `RESULTS.md` row 30, "
    "`quant_sweep_it36l3i8n2i8.json`, `rust_speed_fastrank_on.json`."
)


def raw_sub(data: bytes, old: bytes, new: bytes, label: str) -> bytes:
    n = data.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    print(f"ok [{label}]")
    return data.replace(old, new, 1)


def main():
    res = RES.read_bytes()
    res = raw_sub(res, OLD_TAIL, NEW_TAIL, "RESULTS stale deploy sentence")
    res = raw_sub(res,
                  b"(`rust_fastrank_check.json`, `rust_score.json`) | pending |",
                  b"(`rust_fastrank_check.json`, `rust_score.json`) | pending |\n" + ROW30.encode(),
                  "RESULTS row 30")
    res = raw_sub(res,
                  b"**Row 19 (`nr4`, the note-pool cut) FAILS**",
                  PARA.encode() + b"\n\n**Row 19 (`nr4`, the note-pool cut) FAILS**",
                  "RESULTS row 30 paragraph")
    RES.write_bytes(res)
    print(f"wrote {RES}")

    notes = NOTES.read_bytes()
    notes = raw_sub(notes, b"## Live status\n\n", b"## Live status\n\n" + BULLET.encode() + b"\n\n",
                    "NOTES bullet")
    NOTES.write_bytes(notes)
    print(f"wrote {NOTES}")

    rec = RECORDED.read_bytes()
    if b"it36l3i8n2i8" in rec:
        print("RECORDED.txt already has it36l3i8n2i8")
    else:
        if not rec.endswith(b"\n"):
            rec += b"\n"
        RECORDED.write_bytes(rec + b"it36l3i8n2i8\n")
        print("RECORDED.txt += it36l3i8n2i8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
