"""Chain 7: the equal-bits (rank, precision) probes (2026-11-21 09:30).

Row 15 (card rank-2 int8, 1,536 bits) passes, and `cn8` showed 8-bit precision of the FULL state is free,
so at a FIXED bit count the question is whether the bits should buy RANK or PRECISION. Each point below
has an exact equal-bits counterpart already measured, so the pair is a clean A/B:

  l2i4s4  card rank-2 int4 + int4 shifts =   768 bits   vs row 9  card rank-1 int8 + int4 =   768
  l4i4s4  card rank-4 int4 + int4 shifts = 1,280 bits   vs row 17 card rank-2 int8 + int4 = 1,280
  l7i4s4  card rank-7 int4 + int4 shifts = 2,048 bits   vs row 16 card rank-3 int8 + int8 = 2,048
  n2i4s4  note rank-2 int4 + int4 shifts = 2,304 bits   vs row 17 note rank-1 int8 + int4 = 2,304

The note is held at rank-1 int8 + int4 shifts (2,304) except in `n2i4s4`. Axis 1 (smaller state) is the
top priority, so this is the highest-value axis left: it decides how to spend the bits that remain.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|rwkv-infer")

POINTS = [
    ("l2i4s4", "it36l2i4s4", 124, "qnj_it36ctl"),
    ("l4i4s4", "it36l4i4s4", 124, "qnj_it36ctl"),
    ("l7i4s4", "it36l7i4s4", 124, "qnj_it36ctl"),
    ("n2i4s4", "it36n2i4s4", 124, "qnj_it36ctl"),
]

if __name__ == "__main__":
    sys.exit(lib.chain_main("it36rankprec", "it36notecut_chain.log",
                            "=== it36notecut chain end ===", POINTS, BUSY_PAT))
