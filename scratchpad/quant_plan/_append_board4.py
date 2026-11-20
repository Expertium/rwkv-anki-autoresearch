"""Append Row 11 amendment 4 (the iter36 anchor results) to optimization/GPU_RUN_BOARD.md.

The board carries a BOM, so `edit` cannot match it; append with python instead.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "optimization" / "GPU_RUN_BOARD.md"

blk = """

Row 11 amendment 4 (2026-11-20 09:25, Moscow). **Anchor results: the QAT redeploy is compliant, every
untrained scheme fails.** Everything is on the iter36 anchor, the seeded 750-user roster, paired per
user (`scratchpad/quant_plan/compare_tags.py`), with results in `scratchpad/quant_plan/RESULTS.md`.
Baseline row 0 = `rwkv_iter36_124.pth` unquantized 0.342813 ahead / 0.314272 imm. The other rows:
* control (row 5) = the same 124-step decay replayed in fp32 from `rwkv_iter36_558.pth` on the 5k db
  restricted to users 1-100 (160 groups/epoch, 0 skipped): 0.333802/0.306211, i.e. `-0.009011/-0.008061`.
  The replay's data substitution alone is worth 0.009 -- it is BETTER than iter36's own decay.
* PTQ on the anchor (rows 1-4, `it36ptq`, `quant_sweep_it36ptq.json`): cn2 (2,176/6,528 bits)
  +0.004108/+0.005544; l1i4 (512/1,536) +0.001719/+0.001419; l1i8 (1,024/3,072) +0.001884/+0.001044;
  c2_nl1i8 (2,176/3,072) +0.002085/+0.002054. All FAIL the +0.0005 ceiling by 2.1x-11x. Note that the
  rank-1 family again beats the mixed/full-matrix one at the same base.
* QAT + quantized deploy (row 8, `qc2_nl1i8_it36qatq`) = **0.335038/0.307253 = -0.007776/-0.007019 vs
  the anchor** -> PASS constraint 3 as written, 15x inside the bar, at 2,176/3,072 bits (constraint 2
  satisfied: card exactly at the bar, note half of it).
* QAT with the naive at-the-bar scheme (row 9, `qcn2_it36qatq`) = 0.337559/0.311924: better than the
  anchor but +0.001459/+0.004111 against its own fp32 twin -> QAT does NOT rescue full-matrix int2.
* Tax decomposition vs the budget-matched fp32 control (row 5): PTQ COST +0.001424/+0.002676; MODEL
  DRIFT (QAT trained, evaluated in fp32) +0.002298/+0.001602; PRECISION DEGRADATION (QAT+quant vs its
  own fp32 twin) -0.001063/-0.000560; FULL TAX (QAT+quant vs row 5) **+0.001236/+0.001042**
  (ub95 +0.001375/+0.001205); RECOVERED BY QAT +0.002487/+0.003236. QAT mainly fixes the IMMEDIATE head.
**TWO READINGS OF CONSTRAINT 3, both true:** vs the anchor champion the deploy row is 0.0078/0.0070
better (PASS with a 15x margin); vs its own budget-matched fp32 twin it is +0.0012/+0.0010 worse
(2.4x/2.1x OVER the +0.0005 ceiling). Andrew's decision is needed on which reference the gate uses.
No deploy-contract change is proposed, and the deployed 403 card bits / 214 note bits stay untouched.
Running/queued, one GPU job at a time: `it36qatl1i8` (row 10, candidate `l1i8` at 124 steps, plus its
fp32 twin and the constraint-6 speed run on the idle machine), then `it36qat496` (candidate `c2_nl1i8`
at iter45's 496-step decay length plus its own 496-step fp32 twin -- a scaling diagnostic of the
residual tax), then `it36qatl1i4` (candidate `l1i4` at 124 steps: 512/1,536 bits, four times under the
bar, in case QAT makes the cheaper allocation pass the micro-bar).
"""

with P.open("a", encoding="utf-8") as fh:
    fh.write(blk)
print("lines now:", len(P.read_text(encoding="utf-8").splitlines()))
