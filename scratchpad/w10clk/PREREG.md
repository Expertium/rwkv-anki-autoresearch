# Pre-registration: the leak-free decay branch (`w10clk`), written BEFORE any number exists

Recorded 2026-09-10 22:00. Phase L (the eval-only counterfactual, `scratchpad/leak/PREREG.md`) has
not run yet; arm 2's decay is at step ~13,000 of 21,870. Nothing below is informed by a w10clk or
a phase-L number.

## What w10clk is

ws10's WS-final (`w10_ws_109350.pth`) -> the same 2-epoch (21,870-step) decay as arm 1 -> the same
rectified VAL-half eval, with ONE change: `RWKV_CLOCK_AT_PREV_ANSWER=1800` (rwkv/clock_fix.py) in the
decay, its in-run validation and the eval. Generated from arm 1's runner by `mk_w10clk.py`; the diff
against `run_w10plain.cmd` is the lever, two engagement guards (56: the ON banner, 57: the
chunk-count line that only `apply_to_sample` prints), a non-fatal eval banner post-check, and the
identity strings. Parameters unchanged (563,652). Stub-tested on four paths (clean, no banner,
banner but never ran, eval without banner).

What the fix does: the query row (imm) and the four PAVA probe rows (rectified ahead) predict at
SHOW time, and their clock columns carried the current review's capped-duration excess. When the gap
since the previous answer on any card is below 30 min, those rows are moved to "the card appeared at
the previous answer", and a labelled real row's ahead target time loses the in-session gap of its
target review. Real rows are untouched. `scratchpad/leak/smoke_clock_fix.py` PASSES.

## Why its difference to arm 1 means what it says -- and its one bias

`w10clk - w10plain` is what the clock leak was worth to the metric at a matched training history. It
is a DECAY-ONLY branch off a WS trained WITH the leak, so 2 epochs may not undo all the reliance the
WS learned: **w10clk is a pessimistic estimate of a fully leak-free model**, and its gap to arm 1 is
an UPPER bound on the leak's value once training can adapt. Phase L's lkc30m (arm 1's checkpoint,
unchanged, evaluated leak-free) is the no-adaptation bound above it.

One second difference, not hidden: the fix also shifts the ahead LABEL time (1.64% of ahead targets
move by >= 10% on the raw data; 0.7-2.9% per user in the smoke). Phase L does not. So on AHEAD the
branch includes a channel phase L does not, and the two are not ordered; on IMM they are.

## Predictions

**P1 -- the leak's value after adaptation.** imm worse than arm 1 (0.262066) by **+0.0010..+0.0040**;
ahead worse by **+0.0001..+0.0008**. Reasons: the features bought imm -0.0024 but ahead only -0.0003;
zeroing `t_since_any_review` cost +0.005076 imm / +0.001325 ahead (reliance, not value); the gap
bucket alone carries 0.009 nats on the raw data, most of it in the 11.6% of rows with a 1 s - 30 min
gap.

**P2 -- ordered against phase L on imm.** On users 5001-5300 (the phase-L set), w10clk's imm cost vs
arm 1 is **at most** lkc30m's, and I expect **40-80%** of it. A branch cost ABOVE lkc30m's would mean
the leak-free decay did worse than no adaptation at all -- read it as a defect to find, not a result.

**P3 -- the stop criterion (imm <= 0.2640).** Arm 1 met it with 0.0019 to spare. Prediction, ~60%:
**w10clk does NOT meet it** (P1's middle is ~+0.0025).

**P4 -- the old d=128 model** (0.263586 imm on the same 2,499 users; it has no timestamp features, so
no leak). Prediction, ~55%: **w10clk's imm is worse than d=128's**, i.e. the suspended "beats d=128 on
imm by 0.0015" does not survive. Ahead is unaffected in direction: arm 1 was already 0.003 behind.

**P5 -- engagement, checked by the runner, not after.** The decay log carries the ON banner (56) and
the chunk-count line (57). In the smoke, 32-93% of query rows per user sit in-session (sub-second
gaps included, which move the gap column only) and 3-7% of labels shift.

## What it decides

* **If phase L is MATERIAL (imm >= +0.0010), w10clk is the NEW REFERENCE:** its number replaces arm
  1's as the lineage's honest one, the imm stop criterion and the d=128 comparison are re-read on it,
  and every NEW runner sets `RWKV_CLOCK_AT_PREV_ANSWER`. The relative A/Bs already queued (QAT tax,
  QAT-KD, capacity, budget curve, decay length) keep running: the leak sits in both of their arms.
* **The deploy estimate** becomes w10clk + (arm 2 - arm 1), which assumes the QAT tax is additive
  with the leak fix. Stated, not assumed silently; a leak-free QAT arm is the only direct test.
* **If phase L is below MATERIAL**, w10clk is not inserted ahead of the queue; the fix still becomes
  the default for new runners, because it closes a train/deploy divergence whatever its size.
* **T is not re-tuned on w10clk's number.** If phase L's lkc3h costs >= +0.0005 imm more than
  lkc30m, the branch is regenerated at T = 10800 BEFORE launch (`mk_w10clk.py 10800`), on that
  pre-registered rule, and this file gets a dated note saying so.

**2026-09-10 23:10 -- dated note.** Andrew directed a leak-free run REGARDLESS of phase L (`scratchpad/w10lf/`: WS from scratch with the fix). This branch's predictions stand unchanged; what changes is its role -- it is now a measurement (how much a decay-only fix recovers), not a candidate for the shipped model. It stays armed as-is until phase L reports; if phase L is MATERIAL, the queue is re-planned by hand and this branch may be dropped in favour of the leak-free run. The fix also gained the UTC-midnight calendar shift the same evening (`rwkv/clock_fix.py`), so if this branch runs it uses it.
