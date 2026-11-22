"""Record RESULTS row 46 (`l2i5s4`, 2,624 bits): FAIL - the int5 card costs 6x more per bit.

Sources: scratchpad/quant_plan/quant_sweep_it36l2i5s4.json and it36int6_chain.log.
Writes the RESULTS row + a NOTES live-status bullet (the chain-17 paragraph follows once row 47
lands).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SW46 = RUN / "quant_sweep_it36l2i5s4.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"

A_ROW = b"| 45 |"
A_NOTE = b"## Live status\n\n"
MARK_ROW = b"| 46 |"
MARK_NOTE = b"Row 46 (`l2i5s4`) FAILS on the immediate head"


def f6(x):
    return f"{x:+.6f}"


def main() -> int:
    r = json.loads(SW46.read_text(encoding="utf-8"))["rows"][0]
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    secs = r["secs"]

    row46 = (
        "| 46 | QAT card rank-2 int5 factors + int4 shifts, note rank-1 int6 factors + int4 shifts - "
        "row 45 with the card's factor width cut one more step, int6 -> int5 (896/1,728 = 2,624 bits, "
        "320 below row 45 and 768 below row 35) (`l2i5s4`, `it36int6` point 2, landed 01:22, "
        f"{secs/60:.1f} min; speed cell = row 17's measured `l2i8s4` path - the card's factor width "
        "does not change the verify path and the note stays rank 1) | 896 (112) | 1,728 (216) | 2,290 | "
        f"0.800x | {a['mean_cur']:.6f} | {i['mean_cur']:.6f} | {f6(a['delta'])} | {f6(i['delta'])} | "
        f"**FAIL**: ahead PASSES at 0.70x of the ceiling (ub95 +{a['ub95']:.6f}, p {a['p_worse']:.1e}) "
        f"but the immediate head is 1.13x over it (ub95 +{i['ub95']:.6f}, p {i['p_worse']:.1e}). "
        "Against row 45 (the same note and the same shifts, card at int6) the card's int6 -> int5 cut "
        "costs about +0.000160 ahead / +0.000184 imm for 320 fewer bits - about 6x more per bit than "
        "the int8 -> int6 cut of +0.000073/+0.000054 for 640 - so the card's factor ladder has the "
        "SAME knee at int6 that the note's has (`it36wide6` priced the note's int6 -> int5 step at "
        "+0.000605/+0.000747 for 192 bits), and QAT recovers only about 0.00012/0.00024 of the screen's "
        "PTQ price for this rung (+0.000404/+0.000803 on 7 long histories) | - |"
    )

    raw = RESULTS.read_bytes()
    if MARK_ROW in raw:
        print("RESULTS: row 46 already present")
    else:
        lines = raw.split(b"\n")
        idx = [k for k, l in enumerate(lines) if l.startswith(A_ROW)]
        if len(idx) != 1:
            raise SystemExit(f"ABORT: row anchor found {len(idx)} times")
        k = idx[0]
        eol = b"\r" if lines[k].endswith(b"\r") else b""
        lines[k + 1:k + 1] = [(row46 + (eol.decode() if eol else "")).encode("utf-8")]
        RESULTS.write_bytes(b"\n".join(lines))
        print("RESULTS row 46: ok")

    note46 = (
        "* 2026-11-23 01:30 - **Row 46 (`l2i5s4`) FAILS on the immediate head, so the CARD's factor "
        "ladder also has its knee at int6.** 896/1,728 = 2,624 bits reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {f6(a['delta'])}/{f6(i['delta'])} (ub95 "
        f"+{a['ub95']:.6f}/+{i['ub95']:.6f}): ahead is inside at 0.70x of the ceiling, the immediate "
        "head 1.13x over. The card's int6 -> int5 step costs about +0.000160 ahead / +0.000184 imm for "
        "320 bits, i.e. 6x more per bit than the int8 -> int6 step (+0.000073/+0.000054 for 640), the "
        "same knee the note showed, so int6 is the floor on BOTH streams and 2,624 is not "
        "reachable.\n"
    )

    nb = NOTES.read_bytes()
    if MARK_NOTE in nb:
        print("NOTES: bullet already present")
    else:
        if nb.count(A_NOTE) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(A_NOTE)} times")
        NOTES.write_bytes(nb.replace(A_NOTE, A_NOTE + note46.encode("utf-8"), 1))
        print("NOTES bullet: ok")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())