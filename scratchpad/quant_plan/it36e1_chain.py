"""Chain 15: E1 - the tail-energy penalty, screened on row 17's own recipe (2026-11-22).

WHY. The gate's tax is RANK TRUNCATION, not precision: `cn8` (full-matrix int8, 4x over both bars)
costs only +0.000033/+0.000022. Everything the deploy path loses is the part of the state's energy
that sits BEYOND the stored rank, and the measured size of that part, from row 17's own states
(`corruption/tail_energy_l2i8s4.json`, 5,388 raw states, users 7213/7044/6960) is:

    card, rank 2:  keeps 0.99381 of the state energy  -> tail 0.62%
    note, rank 1:  keeps 0.96943                     -> tail 3.06%

The deploy path stores a rank-r matrix, so a state whose energy is exactly rank-r would lose NOTHING.
The tail is not penalised by the training loss today: the forward pass uses the truncated state, and
the straight-through estimator hands the raw state a gradient as if the truncation were the identity,
so the model has no incentive to keep the tail small. E1 adds the missing signal as
`lambda * mean over streams of (tail energy / total energy)`. It costs ZERO state bits and ZERO deploy
time - it changes training only.

POINTS. `l2i8s4pf` is the control that isolates the fused-off QAT path from the penalty itself (the
fused rank-1 kernel truncates INSIDE the kernel, so a penalty run must use RWKV_QAT_FUSED=0; if the
Python twin is not numerically identical to the fused kernel, that difference would otherwise be
attributed to the penalty). `e1a`/`e1b` bracket lambda at ~2.4% and ~25% of the ~1.02 training loss.
All three are scored against `qnj_it36ctl` - the SAME reference row 17 quotes - so the E1 delta is
read directly off row 17's +0.000052 ahead / +0.000327 imm.

Decision rule: if neither lambda moves the tax, the tail is not reducible at a reasonable cost and E1
is closed (use the tail probe for the mechanism, not the gate numbers, if the tax does not move); if
e1b moves the tax without wrecking the log loss, lambda is refined between 1.0 and 10.0. E1 is also
the only lever here that FREES bits: a smaller truncation tax is what would let the note drop below
rank-1 int8 + int4, which is where the shrink/equalbits/combo chains are pushing.

It runs LAST, after `it36corrupt2`, so it cannot displace any queued chain. BUSY_PAT carries the
corruption driver because a chain queued behind a corruption chain must see that work as GPU work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")

POINTS = [
    ("l2i8s4pf", "it36l2i8s4pf", 124, "qnj_it36ctl"),
    ("l2i8s4e1a", "it36l2i8s4e1a", 124, "qnj_it36ctl"),
    ("l2i8s4e1b", "it36l2i8s4e1b", 124, "qnj_it36ctl"),
]

if __name__ == "__main__":
    sys.exit(lib.chain_main("it36e1", "it36corrupt2_chain.log",
                            "=== it36corrupt2 chain end ===", POINTS, BUSY_PAT))
