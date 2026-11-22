"""Record row 42 (`l3i4s4n3`), the 3,136-bit pre-pricing screen, and chain 21's gated second point.

Sources of truth:
  * scratchpad/quant_plan/quant_sweep_it36l3i4s4n3.json                    - row 42.
  * scratchpad/quant_plan/rust_wide_screen_it36fbwide_b1_it36int6shift.json - the `l2i6s4n1i8s3` price.
  * scratchpad/quant_plan/rust_grid_it36int6shift.json                     - its absolutes.

Writes a RESULTS row + paragraph, two NOTES live-status bullets, GPU_RUN_BOARD Amendment 25 and the
RECORDED.txt stamp.

Run: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_row42.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SW42 = RUN / "quant_sweep_it36l3i4s4n3.json"
WIDE = RUN / "rust_wide_screen_it36fbwide_b1_it36int6shift.json"
GRID = RUN / "rust_grid_it36int6shift.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"

ROW_ANCHOR = b"this exact 3,136 bits) tests | - |"
PARA_ANCHOR = b"unless QAT recovers most of a +0.0020/+0.0028 gap."
STATUS_ANCHOR = b"rows 33-41 FAIL too; row 35 (`n1i8s3`) PASSES** (row 41 "
NOTE_ANCHOR = b"## Live status\n\n"
AMEND = b"## Amendment 25"
MARK = b"**Row 42 (`l3i4s4n3`) closes the 2,368-bit rung"


def f6(x):
    return f"{x:+.6f}"


def main() -> int:
    r = json.loads(SW42.read_text(encoding="utf-8"))["rows"][0]
    g = r["gate"]
    wide = json.loads(WIDE.read_text(encoding="utf-8"))
    w = wide["configs"]["l2i6s4n1i8s3"]
    grid = json.loads(GRID.read_text(encoding="utf-8"))
    gridb = json.loads((RUN / "rust_grid_it36fbwide_b1.json").read_text(encoding="utf-8"))
    n = wide["n_users"]

    def pair(d):
        return f"{f6(d['mean'])} [{f6(d['boot_lo'])}, {f6(d['boot_hi'])}] better {d['better']}/{d['n']}"

    # ---------------------------------------------------------------- RESULTS row 42
    row42 = (
        "| 42 | QAT card **rank-3 int4** factors + int4 shifts, note rank-1 **int3** factors + int4 "
        "shifts - the cheap card under the note's int3 FACTOR rung, 1,024/1,344 = 2,368 bits "
        "(`l3i4s4n3`, `it36combo`, landed 22:43; speed cell = row 33's measured card path `l3i4s4`, "
        "because the card is identical and the note stays rank 1) | 1,024 (128) | 1,344 (168) | "
        "2,290 | 0.800x | "
        f"{g['ahead']['mean_cur']:.6f} | {g['imm']['mean_cur']:.6f} | "
        f"{f6(g['ahead']['delta'])} | {f6(g['imm']['delta'])} | "
        f"**FAIL**: ahead is 1.71x over the ceiling (ub95 +{g['ahead']['ub95']:.6f}) and the immediate "
        f"head 2.69x (ub95 +{g['imm']['ub95']:.6f}, p {g['imm']['p_worse']:.1e} on both). Against row "
        "41 (the SAME card, the note at int8 factors + int3 shifts, 2,112 bits) dropping the note's "
        "factors int8 -> int3, i.e. 768 fewer note bits, costs about +0.00062 ahead / +0.00083 imm "
        "(difference of deltas, not a paired test) - the same order as row 38's +0.00053/+0.00070 for "
        "the same cut on row 17's card, so the note's FACTOR width is the binding constraint a third "
        "time and 2,368 bits is not reachable on this card | - |"
    )

    raw = RESULTS.read_bytes()
    if b"| 42 |" in raw:
        print("RESULTS: row 42 already present")
    else:
        lines = raw.split(b"\n")
        idx = [i for i, l in enumerate(lines) if ROW_ANCHOR in l]
        if len(idx) != 1:
            raise SystemExit(f"ABORT: row anchor found {len(idx)} times, expected 1")
        i = idx[0]
        eol = b"\r" if lines[i].endswith(b"\r") else b""
        lines[i + 1:i + 1] = [(row42 + ("\r" if eol else "")).encode("utf-8")]
        RESULTS.write_bytes(b"\n".join(lines))
        print("RESULTS row 42: ok")

    # ---------------------------------------------------------------- RESULTS paragraph
    para = (
        "**Row 42 (`l3i4s4n3`) closes the 2,368-bit rung, and the 3,136-bit swap is pre-priced as "
        "cheap.** Two results from the same window. (1) The cheap card under the note's int3 FACTOR "
        "rung - card rank-3 int4 + int4 shifts (1,024) with note rank-1 int3 factors + int4 shifts "
        f"(1,344) - lands {g['ahead']['mean_cur']:.6f}/{g['imm']['mean_cur']:.6f} = "
        f"{f6(g['ahead']['delta'])}/{f6(g['imm']['delta'])} (ub95 +{g['ahead']['ub95']:.6f}/"
        f"+{g['imm']['ub95']:.6f}), i.e. 1.71x/2.69x over the ceiling at 2,368 bits. Against row 41 "
        "the 768-bit note factor cut costs about +0.00062/+0.00083 (difference of deltas), the same "
        "order as row 38's +0.00053/+0.00070 for the same cut on row 17's card - so the note's factor "
        "width blocks 2,368 bits whatever card is under it, and the chain's last point "
        "(`l2i4s4n8s3`, 2,880 bits) has the same note at int8 with int3 shifts, which is the only "
        "remaining way to test that size. (2) The 3,136-bit swap that row 41 motivates IS cheap on the "
        f"long histories: `l2i6s4n1i8s3` (the int6 rank-2 card, 1,024 bits, under the note's int3 "
        f"shift rung, 2,112) prices at {pair(w['ahead'])} ahead and {pair(w['imm'])} imm as a per-user "
        f"PAIRED delta against `l2i8s4` on the {n} held-out long histories "
        "(`rust_wide_screen_it36fbwide_b1_it36int6shift.json`; absolutes in "
        f"`rust_grid_it36int6shift.json`: `plain` {grid['plain']['means']['ahead']:.6f}/"
        f"{grid['plain']['means']['imm']:.6f}, `l2i8s4` "
        f"{gridb['l2i8s4']['means']['ahead']:.6f}/{gridb['l2i8s4']['means']['imm']:.6f}, "
        f"`l2i6s4n1i8s3` {grid['l2i6s4n1i8s3']['means']['ahead']:.6f}/"
        f"{grid['l2i6s4n1i8s3']['means']['imm']:.6f}). The imm interval CONTAINS zero and the ahead "
        "interval only just excludes it, so the composition risk (an int6 card AND the note's third "
        "shift bit together) did not materialise - on this deploy-side screen the int6-card version of "
        "3,136 is cheaper on both heads than row 41's rank-3-int4 version, which is why chain 21 runs "
        "it at 750 users. Chain 21 now also carries a GATED second point, `l2i6s4n1i8s2` (the same "
        "card with the note's int2 shifts instead of int3, 1,024/1,920 = 2,944 bits, 448 below row "
        "35), which runs only if the 3,136-bit point PASSES its own gate - the note's int2 shift rung "
        "cost +0.000408 imm in row 20, so it is a coin flip and not worth GPU time unless the rung "
        "above it is proven.\n"
    )

    raw = RESULTS.read_bytes()
    if MARK in raw:
        print("RESULTS: paragraph already present")
    else:
        if raw.count(PARA_ANCHOR) != 1:
            raise SystemExit(f"ABORT: paragraph anchor found {raw.count(PARA_ANCHOR)} times, expected 1")
        RESULTS.write_bytes(raw.replace(PARA_ANCHOR, PARA_ANCHOR + b"\n\n" + para.encode("utf-8"), 1))
        print("RESULTS paragraph: ok")

    # ---------------------------------------------------------------- RESULTS status sentence
    raw = RESULTS.read_bytes()
    if b"rows 33-42 FAIL too" in raw:
        print("RESULTS: status sentence already updated")
    else:
        if raw.count(STATUS_ANCHOR) != 1:
            raise SystemExit(f"ABORT: status anchor found {raw.count(STATUS_ANCHOR)} times, expected 1")
        ins = (b"rows 33-42 FAIL too; row 35 (`n1i8s3`) PASSES** "
               b"(row 42 `l3i4s4n3` +0.000856/+0.001343 - the note's int3 FACTOR rung on the cheap card "
               b"is 1.7x/2.7x over at 2,368 bits, landed 22:43) (row 41 ")
        RESULTS.write_bytes(raw.replace(STATUS_ANCHOR, ins, 1))
        print("RESULTS status sentence: ok")

    # ---------------------------------------------------------------- NOTES bullets
    b42 = (
        "* 2026-11-22 22:50 - **Row 42 (`l3i4s4n3`) closes the 2,368-bit rung: the note's int3 FACTOR "
        "cut is not affordable on ANY card.** Card rank-3 int4 + int4 shifts (1,024) with note rank-1 "
        "int3 factors + int4 shifts (1,344): 0.334658/0.307554 = "
        f"{f6(g['ahead']['delta'])}/{f6(g['imm']['delta'])} (ub95 "
        f"+{g['ahead']['ub95']:.6f}/+{g['imm']['ub95']:.6f}), 1.71x/2.69x over the ceiling. Against "
        "row 41 (same card, note at int8 factors + int3 shifts) the 768-bit factor cut costs about "
        "+0.00062/+0.00083, the same order as row 38's +0.00053/+0.00070 on row 17's card - so the "
        "note's FACTOR width is the binding constraint a third time. `it36combo`'s last point "
        "(`l2i4s4n8s3`, 2,880) keeps the note at int8, so it is the only remaining test of that size; "
        "its decay started 22:45.\n"
    )
    bscr = (
        "* 2026-11-22 22:50 - **The 3,136-bit swap is cheap on the long histories: the int6 card under "
        f"the note's int3 shift rung prices at {f6(w['ahead']['mean'])} ahead / "
        f"{f6(w['imm']['mean'])} imm, imm interval over zero.** `l2i6s4n1i8s3` (1,024/2,112 = 3,136, "
        f"the total row 41 missed by 0.000016) on the {n} held-out long histories, per-user paired vs "
        f"`l2i8s4` (`rust_wide_screen_it36fbwide_b1_it36int6shift.json`): {pair(w['ahead'])} ahead, "
        f"{pair(w['imm'])} imm (range {f6(w['imm']['min'])} to {f6(w['imm']['max'])}). So the "
        "composition risk (int6 card AND the note's third shift bit) did not materialise, and on this "
        "deploy-side screen the int6-card version of 3,136 beats row 41's rank-3-int4 version on both "
        "heads. Chain 21 runs it at 750 users and now carries a GATED second point `l2i6s4n1i8s2` "
        "(note int2 shifts, 1,024/1,920 = 2,944, 448 below row 35) that runs only if the 3,136 point "
        "passes; the chain was restarted at 22:31 to pick the gated list up.\n"
    )

    nb = NOTES.read_bytes()
    if b"Row 42 (`l3i4s4n3`) closes the 2,368-bit rung" in nb:
        print("NOTES: bullets already present")
    else:
        if nb.count(NOTE_ANCHOR) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(NOTE_ANCHOR)} times, expected 1")
        NOTES.write_bytes(nb.replace(NOTE_ANCHOR, NOTE_ANCHOR + b42.encode("utf-8") + bscr.encode("utf-8"), 1))
        print("NOTES bullets: ok")

    # ---------------------------------------------------------------- GPU_RUN_BOARD Amendment 25
    amend = (
        "\n\n## Amendment 25 (2026-11-22 22:50): the note's int3 FACTOR rung blocks 2,368 bits on any "
        "card; the 3,136-bit int6-card swap is pre-priced as cheap; chain 21 gains a gated second rung\n\n"
        "**(1) 2,368 bits is closed.** `l3i4s4n3` (card rank-3 int4 + int4 shifts = 1,024, note rank-1 "
        f"int3 factors + int4 shifts = 1,344) reads {g['ahead']['mean_cur']:.6f}/"
        f"{g['imm']['mean_cur']:.6f} = {f6(g['ahead']['delta'])}/{f6(g['imm']['delta'])} (ub95 "
        f"+{g['ahead']['ub95']:.6f}/+{g['imm']['ub95']:.6f}), i.e. 1.71x/2.69x over the ceiling. "
        "Against row 41 (same card, note at int8 factors + int3 shifts) the 768-bit note factor cut "
        "costs about +0.00062/+0.00083, the same order as row 38's +0.00053/+0.00070 for the same cut "
        "on row 17's card - so the note's FACTOR width blocks that size whatever card sits under it, "
        "and the frontier below 3,136 bits has to come from the note's SHIFT bits, not its factors.\n\n"
        "**(2) The 3,136-bit swap is pre-priced as cheap.** Row 41 reached 3,136 bits with rank-3 int4 "
        "card factors and missed the immediate-head ceiling by 0.000016. The int6 rank-2 card buys the "
        "same 1,024 bits for +0.000052/+0.000115 (Amendment 22), so `l2i6s4n1i8s3` was priced on the "
        f"{n} held-out long histories as a per-user PAIRED delta against `l2i8s4` "
        "(`rust_wide_screen_it36fbwide_b1_it36int6shift.json`, 2,000 bootstrap resamples): "
        f"{pair(w['ahead'])} ahead and {pair(w['imm'])} imm. The imm interval contains zero and the "
        "ahead interval only just excludes it, so stacking the int6 card ON the note's int3 shift rung "
        "does not compound - on this screen it is cheaper on both heads than row 41's card. Chain 21 "
        "measures it at 750 users.\n\n"
        "**(3) Chain 21 now carries a gated second rung.** `l2i6s4n1i8s2` (the same card with the "
        "note's int2 shifts: 1,024/1,920 = 2,944 bits, 448 below row 35) runs ONLY if the 3,136-bit "
        "point passes its own gate, because the note's int2 shift rung cost +0.000408 imm in row 20 "
        "and is therefore a coin flip at best. Both candidates are registered in "
        "`quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and `check_corruption.CONFIGS`; the chain was "
        "restarted at 22:31 so the running process holds the gated list.\n"
    )

    bb = BOARD.read_bytes()
    if AMEND in bb:
        print("BOARD: Amendment 25 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 25 appended")

    # ---------------------------------------------------------------- RECORDED.txt stamp
    rb = RECORDED.read_bytes()
    if b"it36l3i4s4n3" in rb.split(b"\n"):
        print("RECORDED.txt: stamp already present")
    else:
        if not rb.endswith(b"\n"):
            rb += b"\n"
        RECORDED.write_bytes(rb + b"it36l3i4s4n3\n")
        print("RECORDED.txt: added it36l3i4s4n3")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())