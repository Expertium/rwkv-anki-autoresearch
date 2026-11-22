"""Chain 14: RANK 2 (and 3) on the NOTE at int8 factors (Andrew's direction, 2026-11-22 12:01).

Every note cut so far moved the note DOWN in bit width (int4 factors, int3, int2, cheaper shifts) and
failed, while the card's ladder showed that at a fixed budget RANK buys a truncated stream. The note is
the big pool - 2/3 of a passing row's bits - and `n2i4s4` (note rank 2 int4 at the SAME 2,304 bits as
row 17) failed by a hair on the imm head (ub95 +0.000543, 1.09x over the ceiling). So this chain
SPENDS note bits instead:

  l2i8n2i8     card rank-2 int8 + int4 shifts, note rank-2 int8 + int4   1,280 / 3,840 = 5,120
  l3i8n2i8     row 18's card (rank-3 int8), note rank-2 int8 + int4       1,792 / 3,840 = 5,632
  l2i8n2i8s8   as l2i8n2i8 but the note's SHIFT vectors at int8             1,280 / 4,608 = 5,888
  l2i8n3i8     note rank-3 int8 (+ int4 shifts): the rank ladder on the note 1,280 / 5,376 = 6,656

The first point is the direct A/B against row 17: same card, same shifts, only the note's rank changes.
All four are inside the bars (card 2,176, note 6,528) and all are scored against row 0 (`qnj_it36ctl`),
the identical unquantized 124-step decay, like every other row.

NOTE for the record: the deploy engine PARSES and runs a rank >= 2 note. Since 2026-11-22 it also has a
FAST rank-r path (`RWKV_LOWRANK_FAST=1`: 0.803x rev/s for a rank-2 note against 0.618x for the reference
Gram eigensolve) and per-stream shift levels (`CompressCfg::shift_level`), so a passing row here can be
deployed and its speed measured. That work was approved (Andrew, 2026-11-22 12:01: "Sure, try rank-2
note"). See GPU_RUN_BOARD Amendments 19 and 23.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")

POINTS = [
    ("l2i8n2i8", "it36l2i8n2i8", 124, "qnj_it36ctl"),
    ("l3i8n2i8", "it36l3i8n2i8", 124, "qnj_it36ctl"),
    ("l2i8n2i8s8", "it36l2i8n2i8s8", 124, "qnj_it36ctl"),
    ("l2i8n3i8", "it36l2i8n3i8", 124, "qnj_it36ctl"),
]

if __name__ == "__main__":
    sys.exit(lib.chain_main("it36noterank2", "it36corrupt_chain.log",
                            "=== it36corrupt chain end ===", POINTS, BUSY_PAT))
