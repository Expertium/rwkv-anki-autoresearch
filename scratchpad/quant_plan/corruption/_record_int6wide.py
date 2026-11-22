"""Record the 21-user int6/int5 deploy screen (RESULTS paragraph + NOTES bullet + board amendment).

Data: `rust_wide_screen_fast_wide3_int6long_it36int6wide_b4.json`, produced by
`corruption/wide_screen.py` over per-user PAIRED deltas against `l2i8s4`, 2,000 bootstrap resamples.
Batches: `fast_wide3` (users 7250/5052/5939, the 3 longest), `int6long` (the int6/int5 rows on those
same 3 users) and `it36int6wide_b4` (the 18 users the corrupt50 trace export had finished at 19:35).

This is the EVIDENCE BEHIND CHAIN 17 (`it36int6`, the int6/int5/int4 factor ladder), which was queued
on a 2-user short-trace probe that read int6 as free (+0.000003 ahead / +0.000002 imm) and which its
own calibration test had already invalidated (it reproduced a known note tax at only 54% and got a
known card tax's SIGN wrong).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

ANCHOR = (b"measure the tail energy the truncation discards and penalize it, at zero bits and "
          b"zero deploy cost.")

PARA = (
    "\n\n**The factor-precision axis is nearly exhausted: over 21 long-history users an int6 card "
    "costs ~2x what a 3-user probe said, for 640 bits.** Chain 17 asks whether the card's factors can "
    "drop from int8 to int6 (2,944 total bits, 640 below row 17) or int5 (2,624, 960 below), which had "
    "never been TRAINABLE before 2026-11-22 (`rwkv/architecture.py` `_QMAX` lacked int6/int5). Before "
    "spending a 750-user QAT run on it, the deploy engine priced the change on the longest histories "
    "available as per-user PAIRED deltas against `l2i8s4`, with 2,000 bootstrap resamples "
    "(`rust_wide_screen_fast_wide3_int6long_it36int6wide_b4.json`, instrument `corruption/wide_screen.py`, "
    "21 users = the 3 longest plus the 18 the `trace_corrupt50` export had finished). **`l2i6s4` costs "
    "+0.000442 ahead [+0.000232, +0.000701] and +0.000726 imm [+0.000420, +0.001100], better on 2/21 "
    "users on both heads. `l2i5s4` costs +0.001458 ahead [+0.000826, +0.002251] and +0.002189 imm "
    "[+0.001389, +0.003065], better on 1/21.** Both intervals are far clear of zero and the sign is "
    "consistent on 19-20 of 21 users, so this is not an outlier effect. Two readings matter. (1) The "
    "3-user read of the same pair (+0.000345/+0.001551 imm) understated int6 by 2.1x, the same trap "
    "Amendment 17 recorded - the 3-user mean is one hard user plus two others. (2) The PTQ tax is not "
    "the QAT tax: on this sample QAT recovers roughly half of a PTQ tax, but row 17's own QAT imm is "
    "already +0.000327 (ub95 +0.000390) against a +0.0005 ceiling, so int6 would have to recover "
    "almost the entire +0.0007 PTQ gap to land under it. Chain 17 still runs - a roster of 750 is "
    "cheaper than a wrong assumption, and the measured answer is the deliverable - but the expectation "
    "is now on record: **int6 is at best a coin flip and int5 is expected to fail.** CORRECTION: the "
    "screen's last config (`l2i4s4n4`, the 2,304-bit bottom rung) was NOT measured - that name exists "
    "in `quant_sweep`'s candidates but not in `rust_grid`'s CONFIGS, so the batch ended with a "
    "KeyError after the four configs above were complete. With int5 at +0.0022 imm, the bottom rung "
    "has no plausible path, so it was not re-run.\n"
)

BULLET = (
    "* 2026-11-22 19:45 - **int6/int5 priced on 21 long-history users: the factor-precision axis is "
    "nearly exhausted.** Paired deltas vs `l2i8s4`, bootstrap 95%: `l2i6s4` (2,944 bits, 640 below row "
    "17) +0.000442 ahead [+0.000232,+0.000701] / +0.000726 imm [+0.000420,+0.001100], better on only "
    "2/21 - and `l2i5s4` (2,624) +0.001458 / +0.002189, better on 1/21. Sign-consistent on 19-20/21, so "
    "the earlier 3-user read (+0.000345 imm) understated int6 by 2.1x. QAT recovers about half a PTQ "
    "tax on this sample, and row 17's own QAT imm is +0.000327 (ub95 +0.000390) against the +0.0005 "
    "ceiling, so chain 17's int6 point needs nearly the whole gap back: **int6 is at best a coin flip, "
    "int5 is expected to fail.** The screen's bottom rung (`l2i4s4n4`) was not measured (that name is "
    "in `quant_sweep`, not in `rust_grid`'s CONFIGS) and was not re-run - with int5 at +0.0022 imm it "
    "has no plausible path.\n"
)


def sub(raw: bytes, old: bytes, new: bytes, label: str) -> bytes:
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    print(f"  {label}: ok")
    return raw.replace(old, new, 1)


def main() -> int:
    raw = RESULTS.read_bytes()
    if b"The factor-precision axis is nearly exhausted" in raw:
        print("  RESULTS paragraph: already present")
    else:
        raw = sub(raw, ANCHOR, ANCHOR + PARA.encode("utf-8"), "RESULTS paragraph")
        RESULTS.write_bytes(raw)

    notes = NOTES.read_bytes()
    if b"the factor-precision axis is " in notes:
        print("  NOTES bullet: already present")
    else:
        notes = sub(notes, b"## Live status\n\n", b"## Live status\n\n" + BULLET.encode("utf-8"),
                    "NOTES bullet")
        NOTES.write_bytes(notes)
    return 0


BOARD_AMENDMENT = """
## Amendment 20 (2026-11-22 19:45): int6/int5 priced on 21 long histories - chain 17's expectation is on record

Chain 17 (`it36int6`) was queued on a 2-user short-trace PTQ probe that read int6 as free. Its own
calibration test had already invalidated that probe (it reproduced a known note tax at 54% and got a
known card tax's sign wrong), and Amendment 18's method lesson said to route deploy-side probes through
`corruption/wide_screen.py`. That has now been done, on the longest histories available:

| config | bits card/note | total | ahead delta [95%] | imm delta [95%] | better on |
| --- | --- | --- | --- | --- | --- |
| `l2i6s4` | 1,024/1,920 | 2,944 | +0.000442 [+0.000232, +0.000701] | +0.000726 [+0.000420, +0.001100] | 2/21 (both heads) |
| `l2i5s4` | 896/1,728 | 2,624 | +0.001458 [+0.000826, +0.002251] | +0.002189 [+0.001389, +0.003065] | 1/21 (both heads) |

21 users, per-user PAIRED deltas against `l2i8s4` inside each batch, 2,000 bootstrap resamples
(`rust_wide_screen_fast_wide3_int6long_it36int6wide_b4.json`). The sign is consistent on 19-20 of 21
users, so the means are not carried by an outlier. The 3-user read of the same two rungs
(+0.000345/+0.001551 imm) understated int6 by 2.1x.

**What it means for chain 17.** PTQ is not QAT, and QAT recovers roughly half of a PTQ tax on this
sample - but row 17's own QAT imm tax is +0.000327 with ub95 +0.000390 against the +0.0005 ceiling, so
`l2i6s4` would have to win back almost the whole +0.0007 PTQ gap to pass. The chain still runs, because
a 750-user roster is cheaper than a wrong assumption and the measured answer is the deliverable, but the
expectation is now on record: **int6 is at best a coin flip, `l2i5s4` and `l2i4s4n4` are expected to
fail.** The 2,304-bit bottom rung was not measured: `l2i4s4n4` is a `quant_sweep` candidate name that
does not exist in `rust_grid`'s CONFIGS, so that batch ended in a KeyError after the four configs above
were complete; with int5 at +0.0022 imm it was not re-run.
"""


def main_board() -> int:
    board = BOARD.read_bytes()
    if b"## Amendment 20" in board:
        print("  BOARD: Amendment 20 already present")
        return 0
    if not board.endswith(b"\n"):
        board += b"\n"
    BOARD.write_bytes(board + BOARD_AMENDMENT.encode("utf-8"))
    print("  BOARD: Amendment 20 appended")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        rc = main_board()
    sys.exit(rc)
