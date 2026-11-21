"""Chain 9: aggressive shrink combos - push the total state below row 15's 4,608 bits (2026-11-21 09:30).

Row 15 (card rank-2 int8 1,536 / note rank-1 int8 3,072) passes with a lot of headroom on both bars
(card 1,536 of 2,176, note 3,072 of 6,528), so axis 1 has room to cut. These four points trade the two
streams against each other and against the levels, in the region the earlier chains have not touched:

  l3i4s4n4     card rank-3 int4 + int4 shifts, note rank-1 int4 + int4   1,024 / 1,536 = 2,560 total
  l2i8s4n3     card rank-2 int8 + int4 shifts, note rank-1 int3 + int4   1,280 / 1,344 = 2,624 total
  l2i8s2n4     card rank-2 int8 + int2 shifts, note rank-1 int4 + int4   1,152 / 1,536 = 2,688 total
  l3i8s4n4s2   card rank-3 int8 + int4 shifts, note rank-1 int4 + int2   1,792 / 1,152 = 2,944 total

Every one of them is smaller than row 15, and the note is the big pool (2/3 of row 15's bits) so cutting
the note's level is where the total actually moves. Scored against row 0 (the identical unquantized
124-step decay), like every other row.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|rust_speed.py|rust_grid.py|rwkv-infer|"
            "it36qatsmall_chain.py|it36qatsmall2_chain.py|it36qatsmall3_chain.py|"
            "it36qatsmall4_chain.py|it36rank_chain.py|it36notecut_chain.py|"
            "it36rankprec_chain.py|it36longdecay_chain.py")

POINTS = [
    ("l3i4s4n4", "it36l3i4s4n4", 124, "qnj_it36ctl"),
    ("l2i8s4n3", "it36l2i8s4n3", 124, "qnj_it36ctl"),
    ("l2i8s2n4", "it36l2i8s2n4", 124, "qnj_it36ctl"),
    ("l3i8s4n4s2", "it36l3i8s4n4s2", 124, "qnj_it36ctl"),
]

if __name__ == "__main__":
    sys.exit(lib.chain_main("it36shrink", "it36longdecay_chain.log",
                            "=== it36longdecay chain end ===", POINTS, BUSY_PAT))
