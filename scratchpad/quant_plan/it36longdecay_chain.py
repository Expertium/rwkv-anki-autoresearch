"""Chain 8: rank-2 at LONGER decays - the absolute-loss axis (2026-11-21 09:30).

The decay-length axis was closed for the rank-1 tax, but it is the biggest source of ABSOLUTE gain: the
992-step fp32 twin scores 0.317213/0.289632 against row 0's 0.333802/0.306211 (0.0166/0.0166 better), and
the 496-step twin 0.321659/0.294614 (0.0121/0.0116 better). Those gains were invisible while the rank-1
tax ate the budget. With rank-2 the tax is +0.000046/+0.000307, so a longer decay may now buy the absolute
loss while staying inside the ceiling - a better deployed model at the same state size.

Both points use `l2i8` (card rank-2 int8 + int8 shifts, note rank-1 int8) and are scored against their OWN
same-length fp32 twin, which already exists:
  * `it36l2i8496` at 496 steps -> `qnj_it36ctl496` (row 0b)
  * `it36l2i8992` at 992 steps -> `qnj_it36ctl992`

Andrew relaxed the step budget (2026-11-21 01:00): "instead of using 124 steps, you can use more steps ...
if it ultimately ends up turning a failing recipe into a winning one, it's fine". A length mismatch against
the 124-step champion is acceptable but not ideal; each point therefore carries its own same-length twin.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|rwkv-infer")

POINTS = [
    ("l2i8", "it36l2i8496", 496, "qnj_it36ctl496"),
    ("l2i8", "it36l2i8992", 992, "qnj_it36ctl992"),
]

if __name__ == "__main__":
    sys.exit(lib.chain_main("it36longdecay", "it36rankprec_chain.log",
                            "=== it36rankprec chain end ===", POINTS, BUSY_PAT))
