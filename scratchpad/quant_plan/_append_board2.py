# -*- coding: utf-8 -*-
from pathlib import Path

p = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\optimization\GPU_RUN_BOARD.md")
blk = """
Row 11 amendment 2 (2026-11-19 22:35, Moscow). **The iter36 pre/post pair was mapped the other way
round in amendment 1.** The decay phase RESTARTS its step counter, so `rwkv_iter36_558.pth` is the
WS-final (**pre**-decay, the QAT start) and `rwkv_iter36_124.pth` is the post-decay champion (**the
baseline**). Evidence: qat_log #40 describes the lineage as "558 WS + 124 decay"; every decay run is
named after its own step count (iter39/43 = 124, iter44 = 248, iter45 = 496, iter46 = 248); the
opt_iter36 mtimes come in two phases (200-558 written 15:41-15:45, then 50/100/124 written 15:46-15:47)
and the two phases have different optimizer file sizes. So the decay is a **124-step** phase, not
"2 epochs": every step count in this lineage is a multiple of 31, i.e. the old data config had 31
groups/epoch and iter36's decay was 4 of them. The old users-1-100 training dbs (`train_db`,
`train_db_sc8k`) are GONE -- no lmdb survives anywhere on C: or F: except the 5k ones -- so the redo
trains on `train_db_5k_h1_adaptive92` restricted to users 1-100 at MAX_TRAIN_GLOBAL_LEN=66000
(`probe_groups.py`: 160 groups/epoch, 0 chunks dropped, 9.66M rows). Also corrected: iter43/44/45
were NOT QAT runs (HISTORY.md: the restored arch_iter36 was missing the `[QAT]` parser), so their
unquantized scores are valid and they remain legitimate anchor candidates.

Chain launched 22:23 (`scratchpad/quant_plan/it36qat_chain.py`, log `it36qat_chain.log`): it waits
for the anchor baseline eval of `it36_124`, then runs a 0.04-epoch smoke -> the `--no-quant` fp32
control (124 steps) -> the QAT run (candidate `c2_nl1i8` = card full-matrix int2 + note rank-1 int8,
124 steps, 2,176/3,072 bits against the 2,176/6,528 bar) -> an unquantized eval of both new
checkpoints against tag `qnj_it36_124` -> the remaining anchor diagnostics (`it36_558`, `it45_496`,
`it46_248`, `it44_248`). One GPU workload at a time; the chain stops the eval chain itself before
training starts.
"""
with p.open("a", encoding="utf-8") as fh:
    fh.write(blk)
print("lines now:", len(p.read_text(encoding="utf-8").splitlines()))