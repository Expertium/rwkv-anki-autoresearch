"""Record the 21-user error-feedback screen conclusion (Amendment 18 + NOTES bullet).

Screen: row-17 deploy weights, 21 of the longest corrupt50 histories, per-user PAIRED deltas against
`l2i8s4` in the same batch (tags ef_b1/ef_b2/ef_b3 plus the earlier 3-user runs), 2000 bootstrap
resamples for the interval. Raw data: rust_wide_screen_ef_b3_ef_b2_ef_b1_fast_wide3_ef3.json,
rust_grid_ef_b1..b3.json, rust_grid_ef3.json.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTES = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

AMENDMENT = """
## Amendment 18 (2026-11-22 18:30): the error-feedback screen closes - EF is real but cannot shrink the state

The EF question is now answered on **21 of the longest corrupt50 histories** (per-user paired deltas
against `l2i8s4` within each batch, 2,000 bootstrap resamples). Three points, in order of what they
settle:

| config | bits card/note | total | ahead delta [95%] | imm delta [95%] | better on |
| --- | --- | --- | --- | --- | --- |
| `l2i8s4ef` (row 17 + EF) | 1,536/3,072 | 4,608 | -0.001167 [-0.002081, -0.000339] | -0.002453 [-0.004826, -0.000539] | 10/15 ahead, 8/15 imm |
| `l2i4s4n4ef` (both int4 + EF) | 1,024/2,304 | 3,328 | +0.000049 [-0.000737, +0.000875] | -0.000376 [-0.001852, +0.000911] | 10/21 ahead, 6/21 imm |
| `l2i4s4n4ef2` (both int4 + EF, buffer int2) | 896/1,920 | 2,816 | +0.003439 [+0.000587, +0.007513] | +0.002696 [+0.001449, +0.004236] | 1/6 ahead, 0/6 imm |

**(1) EF genuinely works.** At row 17's own config it improves BOTH heads with intervals that exclude
zero, by about -0.0012 ahead and -0.0025 imm. The error buffer is real state, so that costs +1,024 bits
(card +256, the three note layers +768). This is axis 2, and axis 2 is already saturated - rows 29, 30
and 31 sit at or below their own fp32 twins - so the trade is not one the project needs.

**(2) EF cannot shrink the state, which was the whole point of testing it.** The best it does is
`l2i4s4n4ef`: 3,328 bits, i.e. 256 below row 17, reading STATISTICALLY EQUAL to row 17 on both heads
(+0.000049 ahead, -0.000376 imm, both intervals spanning zero, 10/21 users better on ahead and 6/21 on
imm). So int4 factors plus a rank-1 int4 error buffer is a genuine -7% state at equal accuracy, but not
more than that, and it is 7x smaller a cut than the 2,304-bit bottom rung that nobody expects to pass.
Note the direction of the earlier 3-user read: it showed -0.001310 imm and looked like a 256-bit win,
which 21 users cut to -0.000376 and an interval over zero. That is exactly the trap the wide screen
exists to catch.

**(3) A coarser error buffer does not work.** `l2i4s4n4ef2` puts the buffer at int2, which is where the
bits would come from (2,816 bits, 768 below row 17 and exactly row 21's size - the layout that failed by
0.00014). It is WORSE than row 17 on both heads by +0.0034 / +0.0027 with intervals well clear of zero,
and better on 1/6 and 0/6 users. The buffer is the mechanism, so coarsening it removes the benefit. The
engine's `RWKV_EF_PQ` (~96 b) is the only remaining way to cheapen it, and by this result the budget for
the buffer is the whole effect, so it is not worth a chain.

**Disposition.** EF is closed as an axis-1 lever. It stays documented as an axis-2 option that the
current gate cannot score anyway (`RWKV_LOWRANK_EF` exists only in the Rust engine, and the gate is an
eval-path measurement), so it needs no decision from Andrew now - the question only becomes live if a
future recipe wants the -0.0025 imm at more bits.

**Instrument kept.** `corruption/wide_screen.py` aggregates any number of `rust_grid` batch tags into
per-user paired deltas with sign counts and bootstrap intervals, and it is what turned a
three-user-looking win into a measured non-result. Deploy-side probes should go through it rather than
quoting a 2-3 user mean.
"""

BULLET = (
    "* 2026-11-22 18:30 - **The error-feedback screen is closed: EF is real but cannot shrink the "
    "state.** On 21 of the longest corrupt50 histories (per-user PAIRED deltas vs `l2i8s4`, bootstrap "
    "intervals): (a) `l2i8s4ef` (row 17 + EF, 1,536/3,072 = 4,608 bits) improves BOTH heads with "
    "intervals clear of zero, -0.001167 ahead / -0.002453 imm - so the mechanism works - but the buffer "
    "is real state and costs +1,024 bits, which is axis 2 and axis 2 is already saturated (rows 29-31). "
    "(b) `l2i4s4n4ef` (1,024/2,304 = 3,328, i.e. 256 BELOW row 17) is STATISTICALLY EQUAL to row 17 "
    "(+0.000049 ahead / -0.000376 imm, intervals over zero), so EF at int4 factors buys exactly a -7% "
    "state at equal accuracy and no more - the earlier 3-user read of -0.001310 imm shrank to -0.000376 "
    "at 21 users, which is the trap the new wide screen exists to catch. (c) `l2i4s4n4ef2` (buffer at "
    "int2, 896/1,920 = 2,816, exactly row 21's size) is clearly WORSE, +0.003439 ahead / +0.002696 imm "
    "with intervals well clear of zero and better on 1/6 and 0/6 users - the buffer IS the mechanism, so "
    "coarsening it removes the benefit. **EF is closed as an axis-1 lever** and needs no decision from "
    "Andrew unless a future recipe wants the -0.0025 imm at more bits; note it also cannot be gated "
    "today (Rust-only flag, eval-path gate). New instrument: `corruption/wide_screen.py` aggregates "
    "`rust_grid` batch tags into paired deltas with sign counts and intervals.\n"
)


def sub(raw: bytes, old: bytes, new: bytes, label: str) -> bytes:
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    print(f"  {label}: ok")
    return raw.replace(old, new, 1)


def main() -> int:
    notes = NOTES.read_bytes()
    notes = sub(notes, b"## Live status\n\n", b"## Live status\n\n" + BULLET.encode("utf-8"),
                "NOTES bullets")
    NOTES.write_bytes(notes)

    board = BOARD.read_bytes()
    if b"## Amendment 18" in board:
        print("  BOARD: Amendment 18 already present")
    else:
        if not board.endswith(b"\n"):
            board += b"\n"
        BOARD.write_bytes(board + AMENDMENT.encode("utf-8"))
        print("  BOARD: Amendment 18 appended")
    return 0


if __name__ == "__main__":
    sys.exit(main())