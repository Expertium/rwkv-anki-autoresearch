"""Record the `it36wide7` SHIFT-width screen: NOTES bullet, GPU_RUN_BOARD Amendment 33, AGENTS.md.

WHY IT MATTERS. `it36wide7` is the last knob the factor ladder left open: shift width. Its paired
verdict closes it on the immediate head, which makes 2,944 bits (row 45) the floor of the uniform
(rank, factor width, shift width) design space. Recording that stops a future session from spending
GPU hours on a rung the screen already rejects.

Numbers (paired vs the base `l2i6s4` = row 45's layout, 7 held-out users, 2,000 resamples,
`rust_wide_screen_it36wide7.json`):

    l2i6s3n1i6s4  960/1,920 = 2,880  ahead +0.000071 [-0.000082,+0.000245]  imm +0.000497 [+0.000044,+0.001095]
    l2i6s3n1i6s3  960/1,728 = 2,688  ahead +0.000139 [-0.000049,+0.000317]  imm +0.000461 [+0.000031,+0.000993]
    l2i6s2n1i6s3  896/1,728 = 2,624  ahead +0.000848 [+0.000253,+0.001611]  imm +0.002195 [+0.000660,+0.004423]

Line endings: NOTES.md and AGENTS.md are LF; GPU_RUN_BOARD.md is LF as well. The script normalises to LF
internally, verifies every anchor, and writes back in the file's own dominant ending.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]   # scratchpad/quant_plan/corruption -> repo root
NOTES = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
AGENTS = ROOT / "AGENTS.md"

NOTES_ANCHOR = "* 2026-11-23 02:00 - **Row 47 (`l2i4s4n4`) FAILS on both heads"

NOTES_BULLET = """* 2026-11-23 02:45 - **The SHIFT-width knob is CLOSED on the immediate head, so 2,944 bits (row 45) is the floor of the uniform (rank, factor width, shift width) design space.** The paired `it36wide7` deploy screen (base = row 45's own `l2i6s4`, 7 held-out long histories, 2,000 resamples, `rust_wide_screen_it36wide7.json`) prices all three shift rungs: `l2i6s3n1i6s4` (960/1,920 = 2,880, the CARD's shifts int4 -> int3) at **+0.000071 ahead / +0.000497 imm [+0.000044, +0.001095]**, `l2i6s3n1i6s3` (960/1,728 = 2,688, both streams at int3) at **+0.000139 / +0.000461 [+0.000031, +0.000993]**, and `l2i6s2n1i6s3` (896/1,728 = 2,624, the extra card int2 rung) at **+0.000848 / +0.002195 [+0.000660, +0.004423]**. Row 45's immediate head already uses 0.89x of the +0.0005 ceiling (ub95 +0.000446), so a rung that costs +0.00046 imm cannot pass unless QAT recovered ~90% of it - which it does not (the note's int6 cut recovered 9% of its PTQ imm price). So NO QAT point is queued for any shift rung: the shift ladder is closed like the factor ladder (rows 46/47), and both knobs now point at the same 2,944-bit rung. Two consequences. (1) The CARD owns the shift cost, not the note: the both-stream rung costs +0.000461 imm while row 35 priced the NOTE's int4 -> int3 step at only +0.000011/+0.000037, so the note's own 192 shift bits stay cheap. (2) The only live sub-2,944 claim is therefore chain 22's 2,752 point (`l2i6s4n1i6s3` = row 45 minus exactly those 192 note shift bits, PTQ +0.000343/+0.000435), and the axis-1 lever beyond it is no longer bit-cutting but TAX removal at a fixed size (chains 19 fixed-basis, 23/24 E1).

"""

BOARD_AMEND = """
## Amendment 33 (2026-11-23 02:45): the SHIFT-width ladder is closed too - 2,944 bits is the floor of the uniform design space

`it36wide7` is the CPU deploy screen for the last knob the factor ladder left open. Base = row 45's own
`l2i6s4` (1,024/1,920), so these are PAIRED per-user deltas with a 2,000-resample bootstrap, not
differences of marginal means (`rust_wide_screen_it36wide7.json`, `rust_grid_it36wide7.json`):

    config          bits card/note   d ahead                       d imm                      verdict
    l2i6s3n1i6s4      960/1,920      +0.000071 [-0.000082,+0.000245] +0.000497 [+0.000044,+0.001095]  REJECT
    l2i6s3n1i6s3      960/1,728      +0.000139 [-0.000049,+0.000317] +0.000461 [+0.000031,+0.000993]  REJECT
    l2i6s2n1i6s3      896/1,728      +0.000848 [+0.000253,+0.001611] +0.002195 [+0.000660,+0.004423]  REJECT

The ahead head is nearly free on the two int3 rungs (the card's own shift cut is the one that costs),
but the immediate head is not: even the LOW end of each imm interval is at row 45's remaining imm
headroom (row 45 imm = +0.000381, ub95 +0.000446 of the +0.0005 bound, i.e. 0.89x used), and QAT
historically recovers only about a tenth of a PTQ imm price. So no QAT point is spent on any shift rung.

Consequence: the factor ladder closed at int6 (rows 46/47) and the shift ladder closes here, so 2,944
bits (row 45) is the floor of the uniform (rank, factor width, shift width) design space. The remaining
axis-1 claims are (a) chain 22's 2,752-bit `l2i6s4n1i6s3`, which is row 45 minus the NOTE's 192 shift
bits - the note's share of the shift cost, priced by row 35 at +0.000011/+0.000037 - and (b) TAX removal
at a fixed size: the fixed-basis note codec (chain 19) and the E1 tail-energy penalty (chains 23/24).
Row 35's deck of "the note's int4 -> int3 shift step is cheap" survives; what this screen adds is that
the CARD's shift bits are NOT cheap, which is why the both-stream rung costs what the card's rung costs.
"""

AGENTS_START = "measured-cheap knob left is SHIFT width**"
AGENTS_END = "other live sub-3,136 claim (PTQ +0.000343/+0.000435)."

AGENTS_NEW = """SHIFT width is the one remaining knob, and the paired `it36wide7` screen
  (2026-11-23 02:45, base = row 45's own `l2i6s4`) CLOSED it on the immediate head: the CARD's
  int4 -> int3 shift cut costs +0.000071 ahead / +0.000497 imm [+0.000044, +0.001095] (2,880 bits),
  both streams at int3 cost +0.000139/+0.000461 (2,688), and the extra card int2 rung costs
  +0.000848/+0.002195 (2,624). Row 45's imm head already uses 0.89x of the ceiling, so no shift rung
  can pass, and no QAT point is queued for one. **So 2,944 bits (row 45) is the floor of the uniform
  (rank, factor width, shift width) design space.** The last live sub-2,944 claim is chain 22's 2,752
  point (`l2i6s4n1i6s3` = row 45 minus the NOTE's 192 shift bits, which row 35 priced at
  +0.000011/+0.000037 on 750 users); the CARD owns the shift cost, not the note."""


def edit(path: Path, fn) -> bool:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    crlf = text.count("\r\n") > text.count("\n") - text.count("\r\n")
    lf = text.replace("\r\n", "\n")
    new = fn(lf)
    if new is None:
        return False
    n = new.replace("\n", "\r\n") if crlf else new
    path.write_bytes(n.encode("utf-8"))
    print(f"{path.name}: written ({len(raw)} -> {len(n.encode('utf-8'))} bytes, "
          f"{'CRLF' if crlf else 'LF'})")
    return True


def do_notes(lf: str):
    if lf.count(NOTES_ANCHOR) != 1:
        print(f"FAIL: NOTES anchor x{lf.count(NOTES_ANCHOR)}")
        return None
    if "SHIFT-width knob is CLOSED" in lf:
        print("FAIL: NOTES bullet already present")
        return None
    return lf.replace(NOTES_ANCHOR, NOTES_BULLET + NOTES_ANCHOR, 1)


def do_board(lf: str):
    if "Amendment 33" in lf:
        print("FAIL: Amendment 33 already present")
        return None
    if "Amendment 32" not in lf:
        print("FAIL: Amendment 32 missing")
        return None
    return lf.rstrip("\n") + "\n" + BOARD_AMEND


def do_agents(lf: str):
    if AGENTS_START not in lf or AGENTS_END not in lf:
        print("FAIL: AGENTS markers not found")
        return None
    if "design space.** The last live sub-2,944 claim" in lf:
        print("FAIL: AGENTS paragraph already updated")
        return None
    i = lf.index(AGENTS_START)
    j = lf.index(AGENTS_END) + len(AGENTS_END)
    if j - i > 900:
        print(f"FAIL: AGENTS span looks too wide ({j - i} chars)")
        return None
    return lf[:i] + AGENTS_NEW + lf[j:]


def main() -> int:
    ok = True
    for path, fn in ((NOTES, do_notes), (BOARD, do_board), (AGENTS, do_agents)):
        ok = edit(path, fn) and ok
    print("all three written" if ok else "SOMETHING FAILED - check the file above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
