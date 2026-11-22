"""Chain 17: the int6/int5 FACTOR-PRECISION ladder at row 17's exact layout (axis 1, no new code).

WHY THIS IS A NEW LEVER. Row 17 (`l2i8s4`, card rank-2 int8 + int4 shifts, note rank-1 int8 + int4
shifts = 1,280/2,304 = 3,584 bits) is the axis-1 best. EVERY rung below it that has been tested used
int4 factors -- rows 19, 20, 21, 22, 23, 24, 25, 26. The `int6` and `int5` rungs did not exist as
trainable options: `rwkv/architecture.py` `_QMAX` held only int8/int4/int3/int2/fp32, so
`card:2:int6` raised `KeyError` in `_apply_qat_scopes`, even though the Rust deploy engine has always
spoken both levels (`parse_level`, `rust/rwkv-infer/src/model.rs:2104-2105`) and both quantization
paths (fused CUDA and the Python twin) are generic in qmax. The keys were added 2026-11-22 together
with these three candidates, so the ladder is pure axis 1: SAME layout, FEWER bits per factor.

    l2i6s4    1,024/1,920 = 2,944 bits   640 below row 17   (-18%)
    l2i5s4      896/1,728 = 2,624 bits   960 below row 17   (-27%), and 192 below row 21
    l2i4s4n4    768/1,536 = 2,304 bits 1,280 below row 17   (-36%)

WHY IT IS PLAUSIBLE. The one existing measurement that speaks to precision is the old rank-1 deploy
grid (RESULTS.md ~line 349): l1i4/l1i5/l1i6/l1i8 all sat within ~0.0003 of each other, which was read
as "at rank 1 the residual is the RANK TRUNCATION, not the precision". That was rank 1, and PTQ. A
2-user Rust PTQ probe on the row-17 checkpoint (rust_grid_int6_probe.json) says int6 costs
+0.000003 ahead / +0.000002 imm against int8 for 640 fewer bits, and int5 +0.000017 / -0.000073.
l2i4s4n4 is the bottom rung and the only BOTH-int4 combination: row 21 was note-int4 only and row 23
card-int4 only.

WHY THE PROBE IS NOT THE ANSWER. Calibrated against two layouts whose 750-user QAT numbers already
exist, that probe reproduces the note-int8->int4 cost at only 54% and gets the card-int8->int4 SIGN
WRONG. It cannot settle the gate, so every point here is a full 750-user QAT run with the standard
paired test, exactly like rows 15-31.

POINT ORDER is by value: the least aggressive cut first, so a pass is banked before a riskier point is
spent -- l2i6s4 (640 saved) then l2i5s4 (960) then l2i4s4n4 (1,280).

It runs AFTER `it36e1shrink` (chain 16) and shares the queue's BUSY_PAT, including the corruption
driver. It starts nothing itself: it is a waiter, so it costs no GPU until its predecessor's end
marker appears.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")

# (candidate, stamp, steps, ref)
POINTS = [
    ("l2i6s4", "it36l2i6s4", 124, "qnj_it36ctl"),
    ("l2i5s4", "it36l2i5s4", 124, "qnj_it36ctl"),
    ("l2i4s4n4", "it36l2i4s4n4", 124, "qnj_it36ctl"),
]

lib.WAIT_MAX_MIN = 2400      # chains 15 and 16 plus the corruption chain sit ahead of this one

if __name__ == "__main__":
    sys.exit(lib.chain_main("it36int6", "it36e1shrink_chain.log",
                            "=== it36e1shrink chain end ===", POINTS, BUSY_PAT))
