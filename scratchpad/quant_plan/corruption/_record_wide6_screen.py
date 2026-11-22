"""Record the wide6 deploy screen: the note's int5/int4 factor rungs are closed, so 2,752 is the floor.

Sources: scratchpad/quant_plan/rust_wide_screen_it36wide6.json (paired vs `l2i8s4`, 7 users, 2,000
resamples, seed 20261119) and rust_grid_it36wide6.json (the absolutes).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SCREEN = RUN / "rust_wide_screen_it36wide6.json"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

A_NOTE = b"## Live status\n\n"
MARK_NOTE = b"the sub-2,752 corner of this design space is closed"
AMEND = b"## Amendment 31"


def main() -> int:
    s = json.loads(SCREEN.read_text(encoding="utf-8"))["configs"]
    n5, n4 = s["l2i6s4n1i5s3"], s["l2i6s4n1i4s3"]
    a5, i5 = n5["ahead"], n5["imm"]
    a4, i4 = n4["ahead"], n4["imm"]

    bullet = (
        "* 2026-11-23 01:20 - **The note's int5 factor rung is too expensive, so the sub-2,752 corner "
        "of this design space is closed by measurement.** A CPU-only deploy screen (`rust_grid.py "
        "--tag it36wide6`, row-17 deploy weights, the 7 held-out long histories, paired vs `l2i8s4`, "
        "2,000-resample bootstrap) priced the two note-FACTOR rungs under the int6 card: "
        f"`l2i6s4n1i5s3` (1,024/1,536 = 2,560 bits - the same size as the dead-end int2-shift rung, "
        f"bought with the note's factor width instead) reads {a5['mean']:+.6f} "
        f"[{a5['boot_lo']:+.6f},{a5['boot_hi']:+.6f}] ahead / {i5['mean']:+.6f} "
        f"[{i5['boot_lo']:+.6f},{i5['boot_hi']:+.6f}] imm, and `l2i6s4n1i4s3` (2,368 bits) reads "
        f"{a4['mean']:+.6f}/{i4['mean']:+.6f}. So the note's factor ladder has a KNEE at int6: "
        "`it36wide5` priced int8 -> int6 at +0.000170/+0.000216 for 384 bits, while int6 -> int5 "
        "costs about +0.000605/+0.000747 for HALF as many bits (192) - about 6x more per bit. With "
        "row 42's int3 note (1.7x/2.7x over at 2,368) and row 20's int2 note shifts also closed, every "
        "rung below 2,752 is now measured: 2,752 (chain 22's `l2i6s4n1i6s3`, PTQ "
        "+0.000343/+0.000435) is the floor of this design space, and going lower needs TAX removal, "
        "not fewer bits - which is what chains 23/24 (E1) test. The screen reproduces: "
        "`l2i6s4n1i6s3` reads +0.000343/+0.000435 here, exactly its `it36wide5` value.\n"
    )

    nb = NOTES.read_bytes()
    if MARK_NOTE in nb:
        print("NOTES: bullet already present")
    else:
        if nb.count(A_NOTE) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(A_NOTE)} times")
        NOTES.write_bytes(nb.replace(A_NOTE, A_NOTE + bullet.encode("utf-8"), 1))
        print("NOTES bullet: ok")

    amend = (
        "\n\n## Amendment 31 (2026-11-23 01:20): the note's int5 factor rung costs 6x more per bit - the "
        "sub-2,752 corner is closed, so the next gain must come from lower TAX\n\n"
        "A second CPU-only deploy screen (`rust_grid.py --tag it36wide6`, row-17 deploy weights, the "
        "same 7 held-out long histories, paired against `l2i8s4`, 2,000-resample bootstrap, seed "
        "20261119) priced the note's FACTOR width below int6, at the int6 card:\n\n"
        "    config            card/note bits   d ahead                  d imm\n"
        f"    l2i6s4n1i6s3      1,024/1,728      +0.000343 [+0.000229,+0.000492]  "
        f"+0.000435 [+0.000188,+0.000722]   (chain 22's point, unchanged)\n"
        f"    l2i6s4n1i5s3      1,024/1,536      {a5['mean']:+.6f} [{a5['boot_lo']:+.6f},"
        f"{a5['boot_hi']:+.6f}]  {i5['mean']:+.6f} [{i5['boot_lo']:+.6f},{i5['boot_hi']:+.6f}]\n"
        f"    l2i6s4n1i4s3      1,024/1,344      {a4['mean']:+.6f} [{a4['boot_lo']:+.6f},"
        f"{a4['boot_hi']:+.6f}]  {i4['mean']:+.6f} [{i4['boot_lo']:+.6f},{i4['boot_hi']:+.6f}]\n\n"
        "The note's factor ladder has a KNEE at int6. The int8 -> int6 cut costs +0.000170/+0.000216 "
        "for 384 bits (`it36wide5`), but int6 -> int5 costs +0.000605/+0.000747 for 192 bits - about 6x "
        "more per bit - and int4 is 4x further out again. Note also that the int5 rung was bought at "
        "exactly the size of the closed int2-shift rung (2,560 bits): it is BETTER on the immediate "
        "head than that rung (+0.001182 against +0.001642) but 9x worse on ahead (+0.000948 against "
        "+0.000104), and both miss the imm ceiling any way round - so neither way of buying 2,560 bits "
        "works. Every rung below 2,752 is now measured and closed: 2,752 is the floor of "
        "this (rank, factor width, shift width) design space, and the only route further down is "
        "removing deploy TAX at a fixed size - which is what the E1 tail-energy penalty tests in chains "
        "23 and 24.\n\n"
        "Screen artifacts: `rust_grid_it36wide6.json` (absolutes), "
        "`rust_wide_screen_it36wide6.json` (paired deltas + bootstrap), "
        "`rust_grid_it36wide6_stdout.log`. The screen ran beside chain 17 (CPU only, no GPU work) and "
        "was killed before the chain-17 hand-off, as every grid must be.\n"
    )

    bb = BOARD.read_bytes()
    if AMEND in bb:
        print("BOARD: Amendment 31 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 31 appended")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
