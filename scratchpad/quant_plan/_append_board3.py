"""Append Row 11 amendment 3 to optimization/GPU_RUN_BOARD.md.

The board carries a BOM, so `edit` cannot match it; append with python instead.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "optimization" / "GPU_RUN_BOARD.md"

blk = """

Row 11 amendment 3 (2026-11-20 07:47, Moscow). **Anchor redo of iter36 is trained and measured; the
3,000/3,000-bit experiment is VETOED.** Andrew's master spec for this arm caps the card state at half
of naive fp32->int4 (card <= 2,176 bits, note <= 6,528 bits) and he rejected the higher bit budget:
"at that bit budget codebooks become too big + inference speed drops". The "Current constraints" line
below that lists a 3,000/3,000-bit post-queue experiment is therefore superseded -- do NOT launch it.
Anchor = iter36: `rwkv_iter36_558.pth` (WS-final, pre-decay) -> 124-step decay -> `rwkv_iter36_124.pth`
(post-decay, the baseline). All redo runs used `--epochs 2 --max-steps 124 --train-users 1 100
--max-train-len 66000` (5k db restricted to users 1-100: 160 groups/epoch, 0 skipped chunks) from
`rwkv_iter36_558.pth`: smoke `qat_it36qatsmoke` (0.04 ep, rc=0, all three quant banners), fp32 control
`qat_it36ctl` (`--no-quant`, 124 steps, 10.2 min), QAT `qat_it36qat` (candidate `c2_nl1i8` = card
full-matrix int2 + note rank-1 int8, 2,176/3,072 bits, 124 steps, 3.8 min). Unquantized on the seeded
750-user roster, paired per user against `qnj_it36_124` (0.342813 ahead / 0.314272 imm): `it36_558`
0.351102/0.322708 (+0.008289/+0.008436, worse, so the decay does help), `it36ctl` 0.333802/0.306211
(-0.009011/-0.008061), `it36qat` 0.336100/0.307813 (-0.006713/-0.006459). So the 5k-db substitution
gives a BETTER decay than iter36's own, and QAT costs +0.002298 ahead / +0.001602 imm against its
budget-matched fp32 twin. Old-regime diagnostics on the same roster are all 0.036-0.042 behind the
adaptive incumbent (0.300609/0.272009): `it45_496` 0.336628/0.306384, `it46_248` 0.342243/0.313402,
`it44_248` 0.340167/0.311412 -- this lineage is a MECHANISM BED, not a deploy candidate. Running now:
`it36qat_eval2.py` (quantized deploy evals of `qat_124`/`ctl_124`), then `it36ptq_chain.py` (re-base
of the PTQ rows on `rwkv_iter36_124.pth`), then `it36qatl1i8_chain.py` (second QAT screening slot:
candidate `l1i8` = rank-1 int8 on both streams, 1,024/3,072 bits, plus its unquantized twin, its
quantized deploy eval and the constraint-6 speed run on the idle machine). The malformed duplicated
copy of amendment 2 was removed (board was 140 lines, now 118). Results table: `RESULTS.md`.
"""

with P.open("a", encoding="utf-8") as fh:
    fh.write(blk)
print("lines now:", len(P.read_text(encoding="utf-8").splitlines()))
