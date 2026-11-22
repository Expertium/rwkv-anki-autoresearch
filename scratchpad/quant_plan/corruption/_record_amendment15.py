"""Append Amendment 15 to `optimization/GPU_RUN_BOARD.md` (raw bytes; the board's tail is LF).

Covers, for the record: RESULTS row 32 landing and closing the note's rank ladder; `int6`/`int5`
becoming TRAINABLE (the trainer `_QMAX` gap) and chain 17 being queued on that basis; the calibration
that shows the 2-user Rust PTQ probe is a screen and not a gate predictor; and the operational lesson
that a CPU-only job matching a chain's BUSY_PAT strands the queue's handoffs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

AMENDMENT = """
## Amendment 15 (2026-11-22 17:00): row 32 closes the note's rank ladder; `int6`/`int5` become trainable and chain 17 is queued

**RESULTS row 32 is recorded: `l2i8n3i8`, chain 14's last point.** Card rank-2 int8 + int4 shifts
(row 17's card), note **rank-3 int8** + int4 shifts, 1,280/5,376 = 6,656 bits: ahead 0.333817
(+0.000015, ub95 +0.000033, p_worse 0.215), imm 0.306334 (+0.000122, ub95 +0.000142, p_worse 2.0e-30),
**PASS on both heads**. It CLOSES the note's rank ladder: against row 29 (`l2i8n2i8`, the same card)
rank 3 is WORSE on BOTH heads (-0.000115 ahead / +0.000103 imm at rank 2) while spending 1,536 MORE
note bits, and against row 30 (rank 2 on a rank-3 card) it is worse still. So the note's residual tax
is NOT rank-limited - extra note rank cannot buy it back - which is precisely the premise the E1
tail-energy penalty rests on, and keeps chains 15/16 as the right next move. Speed n/a: a rank-3 note
needs `RWKV_LOWRANK_FAST=1` (that pair has not been timed; 0.791x was the rank-2 note on a rank-3 card,
row 30).

**`int6` and `int5` are now TRAINABLE. The factor-precision rungs below int8 had never been tested at
all.** Every rung below row 17 so far used int4 factors (rows 19-26). `int6`/`int5` were not runnable:
`rwkv/architecture.py:151` `_QMAX` carried only int8/int4/int3/int2/fp32, so `card:2:int6` raised
`KeyError` in `_apply_qat_scopes`, even though the Rust deploy engine has ALWAYS spoken both levels
(`parse_level`, `rust/rwkv-infer/src/model.rs:2104-2105`, and `rust_grid.QBITS` - whose docstring has
recorded the gap for a while) and both quantization paths are generic in qmax (`rwkv7_cuda.cu:468-474`
clamps to `+-qmax`; `rwkv_ops._fake_quant_factor` uses `amax/qmax`). The change is therefore three
ADDITIVE edits and no kernel work: `_QMAX` in the trainer, `_QMAX`/`_QBITS` in `quant_sweep` (bits
accounting), and three candidates. No existing recipe names the new levels, so nothing that already ran
changes behaviour. Chain 17 `it36int6` is queued as a waiter behind `it36e1shrink`. All three points sit
at row 17's EXACT layout (card rank 2, note rank 1, int4 shifts), so the ONLY change is factor precision:

| candidate | card/note bits | total | vs row 17 | note |
| --- | --- | --- | --- | --- |
| `l2i6s4` | 1,024/1,920 | 2,944 | 640 fewer | least aggressive cut first |
| `l2i5s4` | 896/1,728 | 2,624 | 960 fewer | also 192 below row 21's 2,816 |
| `l2i4s4n4` | 768/1,536 | 2,304 | 1,280 fewer | the only BOTH-int4 combination (row 21 was note-int4 only, row 23 card-int4 only) |

**A 2-user Rust PTQ probe screens the rungs but cannot settle the gate.** On the row-17 checkpoint,
`rust_grid_int6_probe.json`: `l2i6s4` +0.000003 ahead / +0.000002 imm against `l2i8s4` for 640 fewer
bits; `l2i5s4` +0.000017 / -0.000073 for 960 fewer. That is no visible penalty, but the probe is
calibrated against two layouts whose 750-user QAT numbers already exist, and it FAILS that calibration:
it reproduces the note int8->int4 cost at only 54% (+0.000167 vs the known +0.000311) and gets the card
int8->int4 SIGN wrong (-0.000334 vs the known +0.000253). So every chain-17 point is a full 750-user
QAT run with the standard paired test, exactly like rows 15-31. The probe is only a screen: it shows
the rungs are not catastrophic, nothing more.

**The int6 path was pre-flighted in the REAL trainer on CPU, while the GPU was busy.** With
`DEVICE=cpu` the `[QAT-LOWRANK]` banner reported `card_id=rank2/fq31.0, note_id=rank1/fq31.0` and the
first step trained finite (0.837127 / ahead 0.3954 / imm 0.442), so the level reaches the layer configs
and the trainer runs on it. Chain 17's own 0.04-epoch GPU smoke covers the rest.

**Operational lesson (cost me two jobs and idled the GPU): a CPU-only job that matches a chain's
BUSY_PAT strands the queue.** `it36ladder_lib.running()` matches a process's COMMAND LINE against the
chain's pattern, so `rust_grid.py` (and its `rwkv-infer.exe` children) and even `python -m rwkv.train_rwkv
--config ...qat_*cpufake\\train.toml` on `DEVICE=cpu` both count as "busy". A waiting chain returns only
when `ended AND busy == 0`, so a long CPU job keeps the GPU idle for its whole duration: at 16:42
`it36shrink` had `ended=True` and could not start while `rust_grid.py` was measuring. Both jobs were
killed; the queue took the GPU at 16:46:55 (`it36equalbits`, candidate `l3i4s4`). **Rule: run such jobs
only inside a point's own window** - a point is roughly smoke 1-3 min + decay 10 min + eval 56 min - so
the job must finish well before the point does. The wide-roster fast-solver grid (the 22 landed
`trace_corrupt50` users, `plain/l2i8s4/l2i8s4f/l2i8s4f1`) was launched for that purpose and will be
re-run that way.
"""


def main() -> int:
    raw = BOARD.read_bytes()
    if b"## Amendment 15" in raw:
        print("Amendment 15 already present; nothing to do")
        return 0
    if not raw.endswith(b"\n"):
        raw += b"\n"
    before = len(raw)
    BOARD.write_bytes(raw + AMENDMENT.encode("utf-8"))
    print(f"appended Amendment 15 to {BOARD.name} (+{len(AMENDMENT)} chars, file {before} -> "
          f"{len(BOARD.read_bytes())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
