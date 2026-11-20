"""Append Row 11 amendment 5 (row 10 + constraint-6 speed) to optimization/GPU_RUN_BOARD.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "optimization" / "GPU_RUN_BOARD.md"

blk = """

Row 11 amendment 5 (2026-11-20 10:15, Moscow). **Best recipe so far: QAT with rank-1 int8 on BOTH
streams, and constraint 6 passes.** Row 10 of `RESULTS.md` (`ql1i8_it36qatl1i8q`): QAT 124 steps from
`rwkv_iter36_558.pth` with candidate `l1i8` (card 1,024 bits / note 3,072 bits, both rank-1 int8, 2x
under the card bar and half the note bar), evaluated WITH quantization = **0.334305 ahead / 0.307221
imm** = -0.008508/-0.007052 vs the iter36 anchor champion (ub95 -0.007324/-0.006205). That is the best
row so far: it beats row 8 (`c2_nl1i8`, 2,176/3,072 bits) by 0.000733 ahead / 0.000032 imm while
spending HALF the card bits, and its QAT fp32 twin (0.335448/0.307029) needed less adaptation drift
(+0.001647/+0.000817 vs the control) than the int2-card candidate (+0.002298/+0.001602). On the idle
machine (`rust_speed_it36qatl1i8.json`, 2 reps, users 107/136/156) both recommended configs PASS
constraint 6: `l1i8` 2,586 rev/s = 0.903x / 0.806x states/s / median per-user 0.898, `c2_nl1i8` 2,586 =
0.903x / 0.804x / 0.894, against plain 2,863 rev/s. Quantizer mismatch matters: evaluating the
`l1i8`-trained checkpoint with the `c2_nl1i8` quantizer gives 0.335534/0.308766, i.e. +0.001229/+0.001545
worse than the matched quantizer, so a deploy must use the training-time quantizer.
Queue, one GPU job at a time: `it36qat496` is RUNNING (496-step `c2_nl1i8` QAT + its own 496-step fp32
twin, then fp32 and quantized evals of both: a diagnostic of whether the residual full tax falls with
QAT length), then `it36qatl1i4` (candidate `l1i4` = rank-1 int4, 512/1,536 bits - four times under the
bar - in case QAT makes the cheaper allocation pass the micro-bar). The residual to close is the
immediate head: for row 10 the full tax is +0.000503 ahead (at the ceiling) but +0.001009 imm.
No deploy-contract change is proposed; the deployed 403 card bits / 214 note bits stay untouched.
"""

with P.open("a", encoding="utf-8") as fh:
    fh.write(blk)
print("lines now:", len(P.read_text(encoding="utf-8").splitlines()))
