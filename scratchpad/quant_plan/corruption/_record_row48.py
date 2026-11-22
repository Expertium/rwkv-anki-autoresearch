"""Record RESULTS row 48 (`l2i6c`, chain 18 point 1) + its paragraph, a NOTES bullet and BOARD Amendment 34.

THE RESULT. Card rank-2 int6 + int4 shifts with the note held at row 17's rank-1 int8 - 1,024/2,304 =
3,328 bits, 256 below row 17 - reads 0.333887/0.306575 = +0.000085/+0.000364 against `qnj_it36ctl`
(ub95 +0.000153/+0.000429, p 9.4e-06/0.0, 750 users, 27.2 min). PASS on both heads. Chain 17 cut both
factor widths together, so its ladder could not say WHERE the int6 tax comes from; this point splits it:
the card's int8 -> int6 cut costs +0.000033 ahead / +0.000037 imm for 256 bits (vs row 17's
+0.000052/+0.000327) and the note's costs +0.000040/+0.000017 for 384 bits (vs row 45's
+0.000125/+0.000381). Both are cheap and roughly proportional to the bits they save.

This script also closes the now-stale "the card's shift ladder is the next thing worth pricing" sentence
at the end of the chain-17 paragraph, because the `it36wide7` screen has since priced (and rejected) it.

Line endings: RESULTS.md, NOTES.md and GPU_RUN_BOARD.md are all CRLF in this checkout; the edit is done
in LF and written back as CRLF with a byte round-trip check.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]           # scratchpad/quant_plan/corruption -> repo root
RUN = ROOT / "scratchpad" / "quant_plan"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"

ROW_ANCHOR = ("row 45's 2,944 bits is the frontier and 2,304 was never in reach | - |")
ROW48 = (
    "| 48 | QAT card rank-2 int6 factors + int4 shifts, note rank-1 int8 + int4 shifts - row 17's card "
    "with ONLY the card's factor width cut int8 -> int6, i.e. the CARD half of chain 17's ladder "
    "(1,024/2,304 = 3,328 bits, 256 below row 17) (`l2i6c`, `it36int6card` point 1, landed 02:34, "
    "27.2 min; speed cell = row 17's measured `l2i8s4` path, because the factor width changes only the "
    "quantization LEVELS and the note stays rank-1 int8 with int4 shifts) | 1,024 (128) | 2,304 (288) | "
    "2,290 | 0.800x | 0.333887 | 0.306575 | +0.000085 | +0.000364 | **PASS on both heads**: ahead sits "
    "at 0.31x of the ceiling (ub95 +0.000153, p 9.4e-06) and the immediate head at 0.86x (ub95 "
    "+0.000429, p 0.0), so the card's int8 -> int6 cut costs +0.000033 ahead / +0.000037 imm for 256 "
    "bits while the NOTE's matching cut (row 45 minus this row) costs +0.000040/+0.000017 for 384 bits "
    "- the two cuts are proportional to the bits they save, so neither stream owns the int6 tax | "
    "pending |"
)

PARA_ANCHOR = ("so the card's shift ladder (`l2i6s3n1i6s4` 2,880, `l2i6s3n1i6s3` 2,688) is the next "
               "thing worth pricing before another GPU run.")
PARA_TAIL = (" That pricing is done and the answer is negative: the `it36wide7` screen closed the "
             "card's shift ladder (+0.000071 ahead / +0.000497 imm for the int3 rung, against an "
             "immediate head that row 45 already uses 0.89x of), so 2,944 bits is the floor of the "
             "uniform (rank, factor width, shift width) design space.")

PARA48 = (
    "\n\n**Row 48 (`l2i6c`) passes both heads at 3,328 bits and splits the int6 tax between the two "
    "streams.** Chain 17 cut both factor widths at once, so its ladder could not say where the int6 tax "
    "comes from; `it36int6card` point 1 holds the note at row 17's int8 and cuts only the card, reading "
    "0.333887/0.306575 = +0.000085/+0.000364 (ub95 +0.000153/+0.000429, p 9.4e-06/0.0). Against row 17 "
    "(+0.000052/+0.000327) the card's 256 bits cost +0.000033 ahead / +0.000037 imm; against row 45 "
    "(+0.000125/+0.000381) the note's 384 bits cost +0.000040/+0.000017. So the int6 level is a smooth "
    "trade on EITHER stream, in rough proportion to the bits it saves - unlike int4 and int5, which "
    "break the immediate head outright (rows 46/47) - which is why row 45's 2,944 bits is a stable "
    "frontier rather than a lucky point. Its mirror image, `l2i6n` (card int8 + note int6, 3,200 bits), "
    "lands next. Both rows' finiteness checks are queued in `corruption/_watch_and_check_chain18.py`, "
    "which holds the queue so the checks run in the first free window after the chain's end marker."
)

NOTES_ANCHOR = "* 2026-11-23 02:45 - **The SHIFT-width knob is CLOSED on the immediate head"
NOTES_BULLET = (
    "* 2026-11-23 02:40 - **Row 48 (`l2i6c`) PASSES both heads at 3,328 bits, and the int6 tax turns "
    "out to be split between the streams.** Chain 17's ladder cut both factor widths at once, so a "
    "failure could not say WHERE the int6 tax comes from; `it36int6card` point 1 holds the note at row "
    "17's int8 and cuts only the card (1,024/2,304 = 3,328 bits) and reads 0.333887/0.306575 = "
    "+0.000085/+0.000364 (ub95 +0.000153/+0.000429, p 9.4e-06/0.0). Against row 17 the card's 256 bits "
    "cost +0.000033 ahead / +0.000037 imm; against row 45 the note's 384 bits cost +0.000040/+0.000017 - "
    "cheap on both sides and roughly proportional to the bits saved, unlike int4/int5 which break the "
    "immediate head outright, which is what makes row 45's 2,944 bits a stable frontier rather than a "
    "lucky point. `l2i6n` (card int8 + note int6, 3,200 bits) is the mirror image and lands next. This "
    "row's own finiteness check is queued in `corruption/_watch_and_check_chain18.py` (queue hold up "
    "since 02:08:47), so the RESULTS corruption cell stays `pending` until that check runs.\n"
)

BOARD_AMEND = """
## Amendment 34 (2026-11-23 02:40): chain 18 point 1 (`l2i6c`) PASSES at 3,328 bits - the int6 tax is split between the streams

`it36int6card` took the GPU at 01:59:32, the moment the row-45 finiteness check released it, and its
first point landed at 02:34:05 after 27.2 min: card rank-2 int6 + int4 shifts with the note left at
rank-1 int8 (1,024/2,304 = 3,328 bits, 256 below row 17) reads +0.000085/+0.000364 (ub95
+0.000153/+0.000429, p 9.4e-06/0.0) - a PASS on both heads. It is a diagnostic row, not a frontier row:
it exists to say WHERE the int6 tax comes from. Decomposition against the two ends of chain 17's ladder:
the card's int8 -> int6 cut costs +0.000033 ahead / +0.000037 imm for 256 bits (against row 17) and the
note's costs +0.000040/+0.000017 for 384 bits (against row 45), so the two streams share the int6 tax
roughly in proportion to the bits they save and neither owns it - the opposite of the int4/int5 rungs,
which break the immediate head outright.

Point 2 (`l2i6n`, card rank-1 int8 + note rank-1 int6, 1,280/1,920 = 3,200 bits) started at 02:34:05 and
is the mirror image. The chain has NO corruption phase of its own, so both rows' finiteness checks are
queued in `corruption/_watch_and_check_chain18.py`, which has held the queue since 02:08:47 (a CPU-only
`gpu_hold.py --tag check_corruption`, which every `it36*_chain.py` BUSY_PAT matches) and runs each check
only for a row whose sweep passes, in the first free window after the chain's end marker.
"""


def edit(path: Path, fn):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    crlf = text.count("\r\n") > text.count("\n") - text.count("\r\n")
    lf = text.replace("\r\n", "\n")
    new = fn(lf)
    if new is None:
        return False
    out = new.replace("\n", "\r\n") if crlf else new
    path.write_bytes(out.encode("utf-8"))
    if path.read_bytes().decode("utf-8") != out:
        print(f"FAIL: {path.name} round trip differs")
        return False
    print(f"{path.name}: written ({len(raw)} -> {len(out.encode('utf-8'))} bytes, "
          f"{'CRLF' if crlf else 'LF'})")
    return True


def do_results(lf: str):
    if "| 48 |" in lf:
        print("FAIL: row 48 already present")
        return None
    if lf.count(ROW_ANCHOR) != 1:
        print(f"FAIL: row anchor x{lf.count(ROW_ANCHOR)}")
        return None
    if lf.count(PARA_ANCHOR) != 1:
        print(f"FAIL: paragraph anchor x{lf.count(PARA_ANCHOR)}")
        return None
    out = lf.replace(ROW_ANCHOR, ROW_ANCHOR + "\n" + ROW48, 1)
    out = out.replace(PARA_ANCHOR, PARA_ANCHOR + PARA_TAIL, 1)
    # the new paragraph goes after the chain-17 paragraph that now ends with the it36wide7 sentence
    tail_anchor = PARA_ANCHOR + PARA_TAIL
    out = out.replace(tail_anchor, tail_anchor + PARA48, 1)
    row = [l for l in out.split("\n") if l.startswith("| 48 |")][0]
    if len(row.split("|")) != 14:      # 12 cells -> 14 pieces when split on '|'
        print(f"FAIL: row 48 has {len(row.split('|')) - 2} cells, expected 12")
        return None
    print(f"  row 48 cells: {len(row.split('|')) - 2} (ok)")
    return out


def do_notes(lf: str):
    if "Row 48 (`l2i6c`) PASSES both heads" in lf:
        print("FAIL: NOTES bullet already present")
        return None
    if lf.count(NOTES_ANCHOR) != 1:
        print(f"FAIL: NOTES anchor x{lf.count(NOTES_ANCHOR)}")
        return None
    return lf.replace(NOTES_ANCHOR, NOTES_BULLET + NOTES_ANCHOR, 1)


def do_board(lf: str):
    if "Amendment 34" in lf:
        print("FAIL: Amendment 34 already present")
        return None
    if "Amendment 33" not in lf:
        print("FAIL: Amendment 33 missing")
        return None
    return lf.rstrip("\n") + "\n" + BOARD_AMEND


def do_recorded(lf: str):
    if "it36l2i6c\n" in lf:
        print("FAIL: stamp already present")
        return None
    return lf.rstrip("\n") + "\nit36l2i6c\n"


def main() -> int:
    ok = True
    ok = edit(RESULTS, do_results) and ok
    ok = edit(NOTES, do_notes) and ok
    ok = edit(BOARD, do_board) and ok
    ok = edit(RECORDED, do_recorded) and ok
    print("row 48 recorded" if ok else "SOMETHING FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
