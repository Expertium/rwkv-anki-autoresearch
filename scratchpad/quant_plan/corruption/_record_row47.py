"""Record RESULTS row 47 (`l2i4s4n4`, 2,304 bits): FAIL on both heads - the int4 floor of chain 17.

Sources: scratchpad/quant_plan/quant_sweep_it36l2i4s4n4.json and it36int6_chain.log.
Writes the RESULTS row, the chain-17 paragraph (the ladder is now complete: row 45 PASS at 2,944 with
rows 46-47 failing below it), a NOTES bullet, GPU_RUN_BOARD Amendment 32 and the RECORDED stamp.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SW47 = RUN / "quant_sweep_it36l2i4s4n4.json"
SW45 = RUN / "quant_sweep_it36l2i6s4.json"
SW46 = RUN / "quant_sweep_it36l2i5s4.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"

A_ROW = b"| 46 |"
A_PARA = b"rather than in a later chain."
A_NOTE = b"## Live status\n\n"
MARK_ROW = b"| 47 |"
MARK_PARA = b"**Chain 17 (`it36int6`) settles the int6 factor ladder"
MARK_NOTE = b"Row 47 (`l2i4s4n4`) FAILS on both heads"
AMEND = b"## Amendment 32"


def f6(x):
    return f"{x:+.6f}"


def main() -> int:
    r = json.loads(SW47.read_text(encoding="utf-8"))["rows"][0]
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    secs = r["secs"]
    r45 = json.loads(SW45.read_text(encoding="utf-8"))["rows"][0]["gate"]
    r46 = json.loads(SW46.read_text(encoding="utf-8"))["rows"][0]["gate"]

    # ---------------------------------------------------------------- RESULTS row 47
    row47 = (
        "| 47 | QAT card rank-2 int4 factors + int4 shifts, note rank-1 int4 factors + int4 shifts - "
        "row 45 with BOTH factor widths cut two more steps, int6 -> int4 (768/1,536 = 2,304 bits, 640 "
        "below row 45 and 1,280 below row 17) (`l2i4s4n4`, `it36int6` point 3, landed 01:52, "
        f"{secs/60:.1f} min; speed cell = row 23's measured card path `l2i4s4` - the factor width does "
        "not change the verify path and the note stays rank 1) | 768 (96) | 1,536 (192) | 2,290 | "
        f"0.800x | {a['mean_cur']:.6f} | {i['mean_cur']:.6f} | {f6(a['delta'])} | {f6(i['delta'])} | "
        f"**FAIL on both heads**: ahead is 1.34x over the ceiling (ub95 +{a['ub95']:.6f}, p "
        f"{a['p_worse']:.1e}) and the immediate head 2.25x (ub95 +{i['ub95']:.6f}, p {i['p_worse']:.1e}), "
        "so the int4 floor that rows 23/33 already showed on the int4-rank cards applies to the int6 "
        "family too: row 45's 2,944 bits is the frontier and 2,304 was never in reach | - |"
    )

    raw = RESULTS.read_bytes()
    if MARK_ROW in raw:
        print("RESULTS: row 47 already present")
    else:
        lines = raw.split(b"\n")
        idx = [k for k, l in enumerate(lines) if l.startswith(A_ROW)]
        if len(idx) != 1:
            raise SystemExit(f"ABORT: row anchor found {len(idx)} times")
        k = idx[0]
        eol = b"\r" if lines[k].endswith(b"\r") else b""
        lines[k + 1:k + 1] = [(row47 + (eol.decode() if eol else "")).encode("utf-8")]
        RESULTS.write_bytes(b"\n".join(lines))
        print("RESULTS row 47: ok")

    # ---------------------------------------------------------------- RESULTS paragraph
    para = (
        "**Chain 17 (`it36int6`) settles the int6 factor ladder: row 45 PASSES at 2,944 bits and the "
        "two rungs below it FAIL.** The chain cut row 17's factor widths one step at a time from int8 "
        "to int6 and then further. At int6/int6 (row 45) the whole 640-bit cut costs about +0.000073 "
        f"ahead / +0.000054 imm and PASSES with ub95 +{r45['ahead']['ub95']:.6f}/"
        f"+{r45['imm']['ub95']:.6f}. One more step on the card - int6 -> int5, row 46 - costs about "
        "+0.000160/+0.000184 for 320 bits, i.e. 6x more per bit, and fails the immediate head by 1.13x. "
        "Two steps on both streams - int4/int4, row 47 - fails BOTH heads (1.34x/2.25x over). So int6 "
        "is a knee on both streams, exactly as the CPU deploy screen predicted independently "
        "(`it36wide6`: the note's int6 -> int5 step costs +0.000605/+0.000747 for 192 bits), and the "
        "factor ladder is now closed from below. QAT recovers part of the screen's PTQ price - about "
        "0.00012 ahead / 0.00024 imm at int5/int6 and more at int6/int6 - which is why the QAT runs, "
        "not the screens, decide a rung. What remains below 2,944 is SHIFT width: row 35 measured the "
        "note's int4 -> int3 shift cut at +0.000011/+0.000037 for 192 bits, so the card's shift ladder "
        "(`l2i6s3n1i6s4` 2,880, `l2i6s3n1i6s3` 2,688) is the next thing worth pricing before another "
        "GPU run.\n"
    )

    raw = RESULTS.read_bytes()
    if MARK_PARA in raw:
        print("RESULTS: paragraph already present")
    else:
        if raw.count(A_PARA) != 1:
            raise SystemExit(f"ABORT: paragraph anchor found {raw.count(A_PARA)} times")
        RESULTS.write_bytes(raw.replace(A_PARA, A_PARA + b"\n\n" + para.encode("utf-8"), 1))
        print("RESULTS paragraph: ok")

    # ---------------------------------------------------------------- NOTES bullet
    note47 = (
        "* 2026-11-23 02:00 - **Row 47 (`l2i4s4n4`) FAILS on both heads, so chain 17's ladder is "
        "complete: 2,944 PASSES (row 45), 2,624 and 2,304 fail.** The int4/int4 layout at 768/1,536 = "
        "2,304 bits reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {f6(a['delta'])}/{f6(i['delta'])} (ub95 "
        f"+{a['ub95']:.6f}/+{i['ub95']:.6f}) - 1.34x over on ahead and 2.25x on the immediate head. The "
        "int6 factor level is a knee on BOTH streams (row 46 closed int5, this row closes int4), so the "
        "next cheap knob is SHIFT width, which row 35 measured at +0.000011/+0.000037 for the note's "
        "int4 -> int3 step; the card's shift ladder is being priced on the CPU now "
        "(`l2i6s3n1i6s4` 2,880, `l2i6s3n1i6s3` 2,688).\n"
    )

    nb = NOTES.read_bytes()
    if MARK_NOTE in nb:
        print("NOTES: bullet already present")
    else:
        if nb.count(A_NOTE) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(A_NOTE)} times")
        NOTES.write_bytes(nb.replace(A_NOTE, A_NOTE + note47.encode("utf-8"), 1))
        print("NOTES bullet: ok")

    # ---------------------------------------------------------------- BOARD Amendment 32
    amend = (
        "\n\n## Amendment 32 (2026-11-23 02:00): chain 17 closes the int6 factor ladder - 2,944 PASSES, "
        "2,624 and 2,304 FAIL; the next knob is SHIFT width\n\n"
        "`it36int6` ran three points on row 17's layout, cutting the factor widths from int8 -> 6 and "
        "then below:\n\n"
        "    row  config      card/note bits   d ahead   d imm    verdict\n"
        f"    45   l2i6s4      1,024/1,920      {r45['ahead']['delta']:+.6f}  {r45['imm']['delta']:+.6f}  "
        f"PASS (ub95 {r45['ahead']['ub95']:+.6f}/{r45['imm']['ub95']:+.6f})\n"
        f"    46   l2i5s4        896/1,728      {r46['ahead']['delta']:+.6f}  {r46['imm']['delta']:+.6f}  "
        f"FAIL (imm 1.13x)\n"
        f"    47   l2i4s4n4      768/1,536      {a['delta']:+.6f}  {i['delta']:+.6f}  "
        f"FAIL (1.34x/2.25x)\n\n"
        "So int6 is a knee on both streams: the 640 bits from int8 take +0.000073/+0.000054, the next "
        "320 bits (card int6 -> int5) take +0.000160/+0.000184, and int4 breaks both heads. This "
        "matches the CPU deploy screen run beside the chain (`it36wide6`: the note's int6 -> int5 step "
        "costs +0.000605/+0.000747 for 192 bits), so two independent methods agree on the knee. QAT "
        "does recover part of the PTQ price, which is why these rungs need GPU runs rather than "
        "screens alone.\n\n"
        "Consequence: 2,944 bits (row 45) is the axis-1 frontier, and the one measured-cheap knob left "
        "is SHIFT width (row 35: the note's int4 -> int3 step cost +0.000011/+0.000037 for 192 bits). "
        "A CPU screen (`rust_grid.py --tag it36wide7`) is pricing the card's shift ladder "
        "(`l2i6s3n1i6s4` = 2,880 with only the card's shifts cut, `l2i6s3n1i6s3` = 2,688 with both "
        "streams' shifts at int3) before any further GPU time goes into it. Chain 18 "
        "(`it36int6card`) started as soon as the row-45 finiteness check released the GPU.\n"
    )

    bb = BOARD.read_bytes()
    if AMEND in bb:
        print("BOARD: Amendment 32 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 32 appended")

    # ---------------------------------------------------------------- RECORDED stamp
    rb = RECORDED.read_bytes()
    if b"it36l2i4s4n4" in rb.split(b"\n"):
        print("RECORDED.txt: stamp already present")
    else:
        if not rb.endswith(b"\n"):
            rb += b"\n"
        RECORDED.write_bytes(rb + b"it36l2i4s4n4\n")
        print("RECORDED.txt: added it36l2i4s4n4")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())