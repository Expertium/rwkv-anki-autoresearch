"""Record the `it36wide5` deploy screen: four sub-3,136 rungs priced, two neighbours closed.

Source of truth: scratchpad/quant_plan/rust_wide_screen_it36wide5.json (paired vs `l2i8s4`, 2,000
bootstrap resamples over the 7 held-out long histories) and rust_grid_it36wide5.json (absolutes).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
WIDE = RUN / "rust_wide_screen_it36wide5.json"
GRID = RUN / "rust_grid_it36wide5.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

A_ROW43 = b"so the whole int4-card ladder fails whatever note sits under it | - |"
A_NOTE = b"## Live status\n\n"
MARK = b"## Amendment 28"
PFX = b"**The sub-3,136 corner is priced on the deploy screen"


def f6(x):
    return f"{x:+.6f}"


def pair(d):
    return f"{f6(d['mean'])} [{f6(d['boot_lo'])}, {f6(d['boot_hi'])}] better {d['better']}/{d['n']}"


def main() -> int:
    w = json.loads(WIDE.read_text(encoding="utf-8"))
    g = json.loads(GRID.read_text(encoding="utf-8"))
    # the last grid invocation ran only `plain` plus the four new rungs, so the baseline's absolutes
    # come from the earlier `it36fbwide_b1` grid: SAME weights, SAME traces, SAME 7 users, and the
    # re-run in `rust_grid_it36wide5c_stdout.log` reads 0.407618/0.395728 against its 0.407610/0.395718.
    gb = json.loads((RUN / "rust_grid_it36fbwide_b1.json").read_text(encoding="utf-8"))
    base = g.get("l2i8s4", gb["l2i8s4"])
    n = w["n_users"]
    c = w["configs"]
    a631 = c["l2i6s4n1i8s3"]
    a663 = c["l2i6s4n1i6s3"]
    a531 = c["l2i5s4n1i8s3"]
    a662 = c["l2i6s4n1i6s2"]
    a512 = c["l2i5s4n1i8s2"]

    # ---------------------------------------------------------------- RESULTS paragraph
    para = (
        PFX.decode()
        + f": the int5 card is worse than int6 and the note's int2 shift rung breaks the immediate "
        f"head.** One CPU-only run (`rust_grid.py --tag it36wide5`, the row-17 deploy weights, the same "
        f"{n} held-out long histories every earlier screen used) prices five rungs as per-user PAIRED "
        "deltas against `l2i8s4` "
        f"({base['means']['ahead']:.6f}/{base['means']['imm']:.6f}; `plain` "
        f"{g['plain']['means']['ahead']:.6f}/{g['plain']['means']['imm']:.6f}):\n\n"
        "| config | card/note bits | ahead | imm |\n|---|---|---|---|\n"
        f"| `l2i6s4n1i8s3` (chain 21) | 1,024/2,112 | {pair(a631['ahead'])} | {pair(a631['imm'])} |\n"
        f"| `l2i6s4n1i6s3` (chain 22) | 1,024/1,728 | {pair(a663['ahead'])} | {pair(a663['imm'])} |\n"
        f"| `l2i5s4n1i8s3` | 896/2,112 | {pair(a531['ahead'])} | {pair(a531['imm'])} |\n"
        f"| `l2i6s4n1i6s2` | 1,024/1,536 | {pair(a662['ahead'])} | {pair(a662['imm'])} |\n"
        f"| `l2i5s4n1i8s2` | 896/1,920 | {pair(a512['ahead'])} | {pair(a512['imm'])} |\n\n"
        "Two rungs close on the strength of it. (1) **The int5 card is WORSE than int6**, not a free "
        f"step below it: at the same 2,112-bit note it costs {f6(a531['ahead']['mean'])}/"
        f"{f6(a531['imm']['mean'])} where int6 costs {f6(a631['ahead']['mean'])}/"
        f"{f6(a631['imm']['mean'])}, i.e. about +0.000230 ahead and +0.000580 imm for the last 128 card "
        "bits - 3x the price of every other card bit on offer, and it contradicts the 2-user PTQ probe "
        "that read int5 as free (`rust_grid_int6_probe.json`). So 3,008 bits is not worth a GPU run. "
        "(2) **The note's int2 SHIFT rung is a dead end for the immediate head**: at either factor "
        f"width it costs about +0.0012-0.0014 imm for 192 note bits ({f6(a662['imm']['mean'])} against "
        f"{f6(a663['imm']['mean'])} at int6 factors, {f6(a512['imm']['mean'])} against "
        f"{f6(a531['imm']['mean'])} at int8 factors), which is why chain 21's gated 2,944-bit point is "
        "a long shot.\n\n"
        "The one rung that survives is the note's int6 FACTOR cut stacked on its int3 shift rung: "
        f"`l2i6s4n1i6s3` at 2,752 bits reads {f6(a663['ahead']['mean'])} ahead "
        f"[{f6(a663['ahead']['boot_lo'])}, {f6(a663['ahead']['boot_hi'])}] and "
        f"{f6(a663['imm']['mean'])} imm [{f6(a663['imm']['boot_lo'])}, {f6(a663['imm']['boot_hi'])}]. "
        "The point estimate is inside the +0.0005 ceiling on both heads and it is 640 bits below row 35, "
        "the smallest FULLY deployable row - but the imm upper bound is not inside the ceiling, so it is "
        "marginal by the same rule that rejected row 41's neighbour, while chain 21's 3,136 point sits "
        f"at the ceiling ({f6(a631['imm']['mean'])} [{f6(a631['imm']['boot_lo'])}, "
        f"{f6(a631['imm']['boot_hi'])}]). Chain 22 therefore runs 2,752 as the next size claim and "
        "chain 21's 3,136 runs first. Note the caveat that makes a QAT run necessary: this screen is "
        "PTQ against weights trained for the rank-1-int8 note, so it cannot see what QAT recovers.\n"
    )
    raw = RESULTS.read_bytes()
    if MARK in b"" or PFX in raw:
        print("RESULTS: screen paragraph already present")
    else:
        if raw.count(A_ROW43) != 1:
            raise SystemExit(f"ABORT: A_ROW43 found {raw.count(A_ROW43)} times")
        RESULTS.write_bytes(raw.replace(
            A_ROW43, A_ROW43 + b"\n\n" + para.encode("utf-8"), 1))
        print("RESULTS screen paragraph: ok")

    # ---------------------------------------------------------------- NOTES bullet
    bullet = (
        "* 2026-11-23 00:05 - **The sub-3,136 corner is priced: the int5 card is worse than int6, the "
        "note's int2 shift rung breaks the immediate head, and 2,752 bits is the one survivor.** "
        f"`rust_grid.py --tag it36wide5` over the {n} held-out long histories, paired vs `l2i8s4` "
        f"(2,000 bootstrap resamples; `rust_wide_screen_it36wide5.json`): `l2i6s4n1i8s3` (chain 21, "
        f"3,136) {pair(a631['ahead'])} ahead / {pair(a631['imm'])} imm; `l2i6s4n1i6s3` (chain 22, "
        f"2,752) {pair(a663['ahead'])} / {pair(a663['imm'])}; `l2i5s4n1i8s3` (3,008) "
        f"{pair(a531['ahead'])} / {pair(a531['imm'])}; `l2i6s4n1i6s2` (2,560) "
        f"{pair(a662['ahead'])} / {pair(a662['imm'])}; `l2i5s4n1i8s2` (2,816) "
        f"{pair(a512['ahead'])} / {pair(a512['imm'])}. So int5 costs about +0.00023/+0.00058 versus "
        "int6 for the last 128 card bits (3x every other card bit) and the note's int2 shift rung costs "
        "+0.0012-0.0014 imm at either factor width. `l2i6s4n1i6s3` is the only sub-3,136 rung with its "
        "imm point estimate inside the ceiling, so chain 22 (`it36int6note`) was created at 23:58:57 and "
        "waits behind chain 21; it is registered in all three registries and runs its own corruption "
        "phase if it passes.\n"
    )
    nb = NOTES.read_bytes()
    if b"the int5 card is worse than int6, the note's int2 shift rung" in nb:
        print("NOTES: bullet already present")
    else:
        if nb.count(A_NOTE) != 1:
            raise SystemExit(f"ABORT: A_NOTE found {nb.count(A_NOTE)} times")
        NOTES.write_bytes(nb.replace(A_NOTE, A_NOTE + bullet.encode("utf-8"), 1))
        print("NOTES bullet: ok")

    # ---------------------------------------------------------------- BOARD Amendment 28
    amend = (
        "\n\n## Amendment 28 (2026-11-23 00:05): the sub-3,136 corner is priced - int5 is worse than "
        "int6, the note's int2 shift rung is a dead end, 2,752 bits is the only survivor\n\n"
        f"One CPU-only screen (`rust_grid.py --tag it36wide5`, row-17 deploy weights, the {n} held-out "
        "long histories, paired vs `l2i8s4`, 2,000 bootstrap resamples; "
        "`rust_wide_screen_it36wide5.json`) priced five rungs at once:\n\n"
        "| config | card/note bits | ahead | imm |\n|---|---|---|---|\n"
        f"| `l2i6s4n1i8s3` (chain 21) | 1,024/2,112 | {pair(a631['ahead'])} | {pair(a631['imm'])} |\n"
        f"| `l2i6s4n1i6s3` (chain 22) | 1,024/1,728 | {pair(a663['ahead'])} | {pair(a663['imm'])} |\n"
        f"| `l2i5s4n1i8s3` | 896/2,112 | {pair(a531['ahead'])} | {pair(a531['imm'])} |\n"
        f"| `l2i6s4n1i6s2` | 1,024/1,536 | {pair(a662['ahead'])} | {pair(a662['imm'])} |\n"
        f"| `l2i5s4n1i8s2` | 896/1,920 | {pair(a512['ahead'])} | {pair(a512['imm'])} |\n\n"
        "**(1) The int5 card is worse than int6.** At the same 2,112-bit note the int5 card costs "
        f"{f6(a531['ahead']['mean'])}/{f6(a531['imm']['mean'])} against int6's "
        f"{f6(a631['ahead']['mean'])}/{f6(a631['imm']['mean'])}: about +0.000230/+0.000580 for the last "
        "128 card bits, roughly 3x the price of any other card bit measured so far. The 2-user PTQ probe "
        "that called int5 free (`rust_grid_int6_probe.json`) does not hold on long histories. The 3,008 "
        "rung is therefore not worth a GPU run.\n\n"
        "**(2) The note's int2 SHIFT rung is a dead end for the immediate head.** At int6 factors it "
        f"costs {f6(a662['imm']['mean'])} against {f6(a663['imm']['mean'])} (+0.0012), and at int8 "
        f"factors {f6(a512['imm']['mean'])} against {f6(a531['imm']['mean'])} (+0.0013), for 192 fewer "
        "note bits each time. Chain 21's gated 2,944-bit second point depends on that rung, so it is a "
        "long shot by construction.\n\n"
        "**(3) 2,752 bits is the only survivor, and it is marginal.** `l2i6s4n1i6s3` = card rank-2 int6 "
        "and the note at rank-1 int6 with int3 shifts reads "
        f"{f6(a663['ahead']['mean'])} [{f6(a663['ahead']['boot_lo'])}, "
        f"{f6(a663['ahead']['boot_hi'])}] ahead and {f6(a663['imm']['mean'])} "
        f"[{f6(a663['imm']['boot_lo'])}, {f6(a663['imm']['boot_hi'])}] imm. The point estimate is inside "
        "the +0.0005 ceiling on both heads and 640 bits below row 35, but the imm upper bound is above "
        "the ceiling, while chain 21's 3,136 point reads "
        f"{f6(a631['imm']['mean'])} [{f6(a631['imm']['boot_lo'])}, {f6(a631['imm']['boot_hi'])}] and "
        "sits AT it. So chain 21 runs first and chain 22 (`it36int6note`, created 23:58:57, waiting "
        "behind chain 21) carries the size claim. Both are PTQ prices against weights trained for the "
        "rank-1-int8 note, so neither can see what QAT recovers - which is what the 750-user runs "
        "answer.\n"
    )
    bb = BOARD.read_bytes()
    if MARK in bb:
        print("BOARD: Amendment 28 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 28 appended")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())