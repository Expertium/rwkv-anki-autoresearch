# Pre-registration: `w10qatkd` -- does distilling from the full-precision twin shrink the QAT tax?

Recorded 2026-09-10 13:25, before any GPU time. Andrew, the same day: **"Yep, do KD."**

## The arm, and why its difference from arm 2 is a clean single variable

Arm 2 (`w10qat`) exactly -- ws10's WS-final, the same 2-epoch QAT decay, both catalogs learnable,
evaluated quant-aware with the LEARNED catalogs -- plus `RWKV_QAT_KD=1.0` with
`RWKV_QAT_KD_TEACHER=scratchpad/ws10/w10p_d_21870.pth`, i.e. **arm 1's decayed final: the same
model, same WS, same decay batches, in full precision.** Verified by diffing the env in force at the
decay call: the two arms differ in exactly those two variables (plus paths and tags).
So **`w10qat - w10qatkd` is what KD does to the QAT tax**, and `w10qatkd - w10plain` is the tax
that remains.

**Form.** The live teacher forwards the same prepared batch each step (no dump, so alignment holds
by construction). The term is additive, `hard + lambda * (soft CE on the rating head + soft BCE on
the curve head)`, which for targets that enter linearly is the target mix `alpha = lambda/(1+lambda)
= 0.5` -- the established decay-KD strength (iter 45). ⚠ One difference from iter 45's dump KD: the
dump path rewrote `label_y` and so also softened the PAVA probe targets; the additive path leaves
the probe path on hard labels.

## ⚠ The live path had a bug; it was fixed and proven before this pre-registration

`qat_kd_targets` (rwkv/train_rwkv.py) built the teacher's CURVE target with the plain
`forgetting_curve(w, t)` for every model, while under `RWKV_GRU_HEAD` the student's own curve is
`gru_forgetting_curve(w, s, d, t)`. `scratchpad/qatkd/smoke_qatkd.py` (CPU, real model, real gen-5
chunk, calling the code the loop calls):

| check | result |
|---|---|
| stripped QAT-env teacher vs the plain model | IDENTICAL, 0.0e+00 (9 hooks stripped) |
| unstripped QAT model vs plain (non-vacuity) | differs by up to 1.08 |
| KD gradient at teacher == student, FIXED target | 2.9e-07 of the hard-loss gradient (vanishes) |
| same, the PRE-FIX target | 0.79 of the hard-loss gradient |

## Predictions, recorded before the run

* **P1 (the question).** KD shrinks the tax in BOTH modes: `w10qatkd` beats arm 2 by >= +0.0001 in
  each at paired p < 1e-4. Expected size, if arm 2's tax lands near the 1.25-ep prior
  (+0.0023 / +0.0035): **10-30% of it**, i.e. +0.0002..+0.0007 ahead, +0.0003..+0.0010 imm. The
  ceiling is the tax itself -- a quantized student distilled toward its fp twin cannot pass the twin.
* **P2.** The absolute recovery is larger on imm, because imm carries the larger tax. Weak.
* **P3 (engagement, checkable in the first minutes).** The decay log names the teacher (runner
  guard 81), and the KD run's training loss sits ABOVE arm 2's at the same step by roughly
  lambda x the teacher's entropy terms -- an additive soft term with a non-degenerate teacher
  cannot be zero. If the two loss traces coincide, the term is not reaching the loss.
* **P4 (falsifier).** Both modes within +/-0.0001 of arm 2, with P3 satisfied: distillation from
  the fp twin does not shrink the tax at this budget, i.e. the tax is not a drift a teacher can
  anchor. Then the remaining zero-byte axis is the WKV/shift bit split (w10qat/PREREG.md).

## Gate

The BOTH-MODES rule, not the curve-side exception: this KD rewrites both objectives (the iter 55
note in CLAUDE.md). Adopt KD into the deploy QAT recipe iff P1 holds.

## Cost

~27 h decay (arm 2's ~0.24 steps/s plus one bf16 no-grad teacher forward per step) + ~10.5 h for
the probe and the quant-aware eval. Queued after cmixgraft, before the budget curve.
