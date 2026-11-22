"""Record row 43 (`l2i4s4n8s3`), the last point of `it36combo`, plus the screen reproduction.

Sources of truth:
  * scratchpad/quant_plan/quant_sweep_it36l2i4s4n8s3.json  - row 43's 750-user sweep.
  * scratchpad/quant_plan/rust_grid_preds/it36wide5/*       - the 7-user screen re-run (CPU).

Writes a RESULTS row + paragraph, two NOTES live-status bullets, GPU_RUN_BOARD Amendment 26 and the
RECORDED.txt stamp.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SW43 = RUN / "quant_sweep_it36l2i4s4n8s3.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"

A_ROW = (b"so the note's FACTOR width is the binding constraint a third time and 2,368 bits is not "
         b"reachable on this card | - |")
A_PARA = b"unless the rung above it is proven."
A_STAT = b"rows 33-42 FAIL too; row 35 (`n1i8s3`) PASSES** (row 42 "
A_NOTE = b"## Live status\n\n"
MARK = b"**Row 43 (`l2i4s4n8s3`) closes the int4-card ladder"
AMEND = b"## Amendment 26"


def f6(x):
    return f"{x:+.6f}"


def main() -> int:
    r = json.loads(SW43.read_text(encoding="utf-8"))["rows"][0]
    g = r["gate"]

    # ---------------------------------------------------------------- RESULTS row 43
    row43 = (
        "| 43 | QAT card **rank-2 int4** factors + int4 shifts, note rank-1 int8 + **int3** shifts - "
        "the int4 card under the cheap note, 768/2,112 = 2,880 bits (`l2i4s4n8s3`, `it36combo`'s LAST "
        "point, landed 23:17, 25.1 min; speed cell = row 23's measured card path `l2i4s4`, because the "
        "card is identical) | 768 (96) | 2,112 (264) | 2,290 | 0.800x | "
        f"{g['ahead']['mean_cur']:.6f} | {g['imm']['mean_cur']:.6f} | "
        f"{f6(g['ahead']['delta'])} | {f6(g['imm']['delta'])} | "
        f"**FAIL**: ahead PASSES at 0.70x of the ceiling (ub95 +{g['ahead']['ub95']:.6f}, p "
        f"{g['ahead']['p_worse']:.1e}) but the immediate head is 1.22x over (ub95 "
        f"+{g['imm']['ub95']:.6f}, p {g['imm']['p_worse']:.1e}). Row 23 is this card's int4-shift twin "
        "at 3,072 bits (+0.000266/+0.000580), so the note's int3 SHIFT rung costs only "
        "+0.000015/+0.000031 for 192 fewer note bits - and the imm head is over in BOTH, so the whole "
        "int4-card ladder fails whatever note sits under it | - |"
    )

    raw = RESULTS.read_bytes()
    if b"| 43 |" in raw:
        print("RESULTS: row 43 already present")
    else:
        lines = raw.split(b"\n")
        idx = [i for i, l in enumerate(lines) if A_ROW in l]
        if len(idx) != 1:
            raise SystemExit(f"ABORT: row anchor found {len(idx)} times")
        i = idx[0]
        eol = b"\r" if lines[i].endswith(b"\r") else b""
        lines[i + 1:i + 1] = [(row43 + (eol.decode() if eol else "")).encode("utf-8")]
        RESULTS.write_bytes(b"\n".join(lines))
        print("RESULTS row 43: ok")

    # ---------------------------------------------------------------- RESULTS paragraph
    para = (
        "**Row 43 (`l2i4s4n8s3`) closes the int4-card ladder, and the deploy screen reproduces to "
        "1e-5.** Two results from the same window. (1) The 2,880-bit rung - the int4 card (768) under "
        "the note's int3 SHIFT rung (2,112), i.e. row 23's card with 192 fewer note bits - reads "
        f"{g['ahead']['mean_cur']:.6f}/{g['imm']['mean_cur']:.6f} = {f6(g['ahead']['delta'])}/"
        f"{f6(g['imm']['delta'])} (ub95 +{g['ahead']['ub95']:.6f}/+{g['imm']['ub95']:.6f}): the ahead "
        "head passes at 0.70x of the ceiling, the immediate head is 1.22x over. Row 23 (the same card, "
        "the note at int4 shifts, 3,072 bits) is 1.16x over on the same head, so the note's int3 shift "
        "rung is nearly free (+0.000015/+0.000031 for 192 bits) and the LIMIT is the card: at rank 2 "
        "int4 costs the imm head about +0.00039 against row 17 and neither 2,880 nor 3,072 survives it. "
        "(2) The 7-user CPU screen was re-run from scratch on the row-17 deploy weights "
        "(`rust_grid.py --tag it36wide5`) and reproduces the earlier numbers: `plain` "
        "0.411068/0.398831 against 0.411063/0.398798, `l2i8s4` 0.407618/0.395728 against "
        "0.407610/0.395718, and chain 21's point 1 `l2i6s4n1i8s3` 0.407791/0.395947, i.e. "
        "+0.000173/+0.000219 against the earlier +0.000164/+0.000205. So the screen is reproducible to "
        "about 1e-5 and its prices can be trusted to that level - which matters because it is the only "
        "thing standing between a cheap CPU probe and a 25-minute GPU run.\n"
    )

    raw = RESULTS.read_bytes()
    if MARK in raw:
        print("RESULTS: paragraph already present")
    else:
        if raw.count(A_PARA) != 1:
            raise SystemExit(f"ABORT: paragraph anchor found {raw.count(A_PARA)} times")
        RESULTS.write_bytes(raw.replace(A_PARA, A_PARA + b"\n\n" + para.encode("utf-8"), 1))
        print("RESULTS paragraph: ok")

    # ---------------------------------------------------------------- status sentence
    raw = RESULTS.read_bytes()
    if b"rows 33-43 FAIL too" in raw:
        print("RESULTS: status sentence already updated")
    else:
        if raw.count(A_STAT) != 1:
            raise SystemExit(f"ABORT: status anchor found {raw.count(A_STAT)} times")
        ins = (b"rows 33-43 FAIL too; row 35 (`n1i8s3`) PASSES** "
               b"(row 43 `l2i4s4n8s3` +0.000281/+0.000611 - the int4 card's LAST rung, 2,880 bits, dies "
               b"on the immediate head at 1.22x, landed 23:17) (row 42 ")
        RESULTS.write_bytes(raw.replace(A_STAT, ins, 1))
        print("RESULTS status sentence: ok")

    # ---------------------------------------------------------------- NOTES bullets
    b43 = (
        "* 2026-11-22 23:20 - **Row 43 (`l2i4s4n8s3`, 2,880 bits) closes the int4-card ladder: ahead "
        "passes, imm fails.** `it36combo`'s last point - the int4 card (768) under the note's int3 "
        f"shift rung (2,112) - reads 0.334083/0.306823 = {f6(g['ahead']['delta'])}/"
        f"{f6(g['imm']['delta'])} (ub95 +{g['ahead']['ub95']:.6f}/+{g['imm']['ub95']:.6f}): ahead is "
        "0.70x of the ceiling, imm 1.22x over. Against row 23 (same card, note int4 shifts, 3,072) the "
        "note's int3 shift rung costs only +0.000015/+0.000031 for 192 bits, so the note is NOT the "
        "obstacle here and the int4 CARD is: it costs the imm head about +0.00039 versus row 17. "
        "`it36combo` logged `=== it36combo chain end ===` at 23:17:03, its list is exhausted, and "
        "`it36corrupt2` then skipped `l2i4s4`, `l3i4s4` and `l2i8s2` (all three fail the loss gate, so "
        "they are never deploy candidates) and started the 50-user finiteness check for `n1i8s3` (row "
        "35, the smallest passing row) at 23:21:17.\n"
    )
    bscr = (
        "* 2026-11-22 23:20 - **The 7-user CPU screen reproduces to 1e-5, and the sub-3,136 corner is "
        "now priced on it.** `rust_grid.py --tag it36wide5` re-ran the row-17 deploy weights over the "
        "same 7 held-out long histories: `plain` 0.411068/0.398831 (earlier 0.411063/0.398798), "
        "`l2i8s4` 0.407618/0.395728 (earlier 0.407610/0.395718), `l2i6s4n1i8s3` 0.407791/0.395947 = "
        "+0.000173/+0.000219 (earlier +0.000164/+0.000205). So a screen price is good to about 1e-5 and "
        "chain 21's point 1 stands as pre-priced cheap. Two new rungs are registered in "
        "`rust_grid.CONFIGS` for the same screen - `l2i5s4n1i8s3` (896/2,112 = 3,008, the int5 card "
        "under the note's int3 shift rung) and `l2i5s4n1i8s2` (896/1,920 = 2,816) - and the 2,880 rung "
        "is CLOSED by row 43, so below 3,136 bits the only levers left are the card's factor width "
        "between int5 and int6 and the note's int2 shift bit.\n"
    )

    nb = NOTES.read_bytes()
    if b"Row 43 (`l2i4s4n8s3`, 2,880 bits) closes the int4-card ladder" in nb:
        print("NOTES: bullets already present")
    else:
        if nb.count(A_NOTE) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(A_NOTE)} times")
        NOTES.write_bytes(nb.replace(A_NOTE, A_NOTE + b43.encode("utf-8") + bscr.encode("utf-8"), 1))
        print("NOTES bullets: ok")

    # ---------------------------------------------------------------- BOARD Amendment 26
    amend = (
        "\n\n## Amendment 26 (2026-11-22 23:20): 2,880 bits is closed, `it36combo` is exhausted, and "
        "the CPU screen reproduces to 1e-5\n\n"
        "**(1) 2,880 bits is closed.** `l2i4s4n8s3` (card rank-2 int4 + int4 shifts = 768, note rank-1 "
        f"int8 + int3 shifts = 2,112) reads {g['ahead']['mean_cur']:.6f}/{g['imm']['mean_cur']:.6f} = "
        f"{f6(g['ahead']['delta'])}/{f6(g['imm']['delta'])} (ub95 +{g['ahead']['ub95']:.6f}/"
        f"+{g['imm']['ub95']:.6f}). The ahead head passes at 0.70x of the ceiling; the immediate head is "
        "1.22x over. It was `it36combo`'s last point.\n\n"
        "**(2) The int4-card ladder is closed as a whole, and the CARD is what closes it.** Row 23 (the "
        "same card, note int4 shifts, 3,072 bits) is 1.16x over on the immediate head; row 43 is 1.22x "
        "over. The note's int3 shift rung therefore costs only +0.000015/+0.000031 for 192 note bits, "
        "while the step from row 17's int8 card to an int4 card costs the imm head about +0.00039. So "
        "the reduction below 3,136 bits has to come from the card's factor width (int6 -> int5) or from "
        "the note's int2 shift bit, not from the note's factors or its rank-1 int8 precision.\n\n"
        "**(3) The 7-user CPU screen is reproducible to 1e-5.** `rust_grid.py --tag it36wide5` re-ran "
        "the row-17 deploy weights over the same 7 held-out long histories: `plain` 0.411068/0.398831 "
        "(earlier 0.411063/0.398798), `l2i8s4` 0.407618/0.395728 (earlier 0.407610/0.395718), "
        "`l2i6s4n1i8s3` 0.407791/0.395947 = +0.000173/+0.000219 (earlier +0.000164/+0.000205). Two new "
        "rungs are registered for it in `rust_grid.CONFIGS`: `l2i5s4n1i8s3` (3,008) and `l2i5s4n1i8s2` "
        "(2,816).\n\n"
        "**(4) Chain state.** `it36combo` ended at 23:17:03 with its list exhausted; `it36corrupt2` "
        "skipped the three gate-failing points and started the 50-user check for `n1i8s3` (row 35) at "
        "23:21:17. The remaining waiters are `it36e1shrink`, `it36fixedbasis`, `it36e1`, `it36int6card`, "
        "`it36int6combo` and `it36int6`.\n"
    )

    bb = BOARD.read_bytes()
    if AMEND in bb:
        print("BOARD: Amendment 26 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 26 appended")

    # ---------------------------------------------------------------- RECORDED stamp
    rb = RECORDED.read_bytes()
    if b"it36l2i4s4n8s3" in rb.split(b"\n"):
        print("RECORDED.txt: stamp already present")
    else:
        if not rb.endswith(b"\n"):
            rb += b"\n"
        RECORDED.write_bytes(rb + b"it36l2i4s4n8s3\n")
        print("RECORDED.txt: added it36l2i4s4n8s3")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
