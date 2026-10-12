# WS TUNING AT 332k -- the plan (Andrew 2026-09-22, knobs and order delegated to me)

> Andrew: *"WS three with full trials and early pruning of bad runs at 3 epochs, got it."* and
> *"I'll leave the details of which hyperparameters to tune and in which order to you."*

Phase 5 as written tunes only the QAT decay (his 2026-09-12 scope decision). This adds the WS phase,
which has never been tuned at 10 epochs or at this size -- the champion HPs were set at d=32 and
confirmed at 558k on a 1.25-epoch budget.

**Shrinking is what made this affordable.** A WS trial was ~38 h at 561k, which is exactly why phase
5 was scoped to the decay. At 332k it is **31.1 h**, and ~20 h averaged once losers are pruned.

## ⚠ CORRECTION 2026-09-29: LR KNOBS CANNOT BE RANKED ON THE WS ENDPOINT

Under WSD the WS phase runs at a constant LR, and a LOWER LR sits lower in the loss at the end of the WS
because it carries less optimizer noise; the decay removes most of that noise for the higher-LR run. So a WS
endpoint is biased toward the smaller LR, and the bias is exactly what the knob changes. Measured on the two LR
trials (window mean of the last 5 validations vs s64hc10, + = worse):

| trial | ahead | imm |
|---|---|---|
| wtlr2 (LR x2) | +0.00022 | +0.00061 |
| wtlr0p5 (LR x0.5) | -0.00110 | +0.00024 |

**Rule: an LR-type trial (lr, muon) buys its 1-epoch decay + full-VAL eval before any verdict**, and the gate is
the either-mode rule against 104 `s64hc10`. wtlr2's WS-endpoint rejection is withdrawn and re-opened. The endpoint
rule still serves warmup (the LR at the WS end is the same) and the epoch-3 kill (it only removes disasters).
A second bug made this worse: `mk_ws_trial.py` passed a hardcoded decay peak LR of 1e-3 to `write_decay_setup`
even when the knob changed PEAK_LR (fixed; the decays are bought by `scratchpad/wstune/mk_decay_eval.py`,
which reads the LR from the trial's ws.toml and guards it in decay.toml and in the decay log).
Chain v3 (`scratchpad/wstune/chain_ws3.cmd`): wtlr0p5 decay+eval, wtlr2 decay+eval, then the d128e10 waiter.
The Muon and warmup trials are regenerated on whichever LR wins, and run after d128e10.

## The design

**Full 10-epoch trials, losers killed at epoch 3.** NOT shortened trials: under WSD a 3-epoch run
has a different LR trajectory than a 10-epoch one, and LR is the main knob -- so a shortened trial
is unreliable exactly where it would be used most. Killing a loser costs 9.3 h instead of 31.1 h and
costs nothing in validity, because every survivor runs the real schedule.

**Ranking instrument: the validation endpoint, zero GPU cost.** A run's 10-user validation endpoint
read beside the incumbent's on the SAME users predicted arm 1's gate deltas within 0.0004. A trial
needs a decay + full eval only when it is about to be ADOPTED.

**⚠ The epoch-3 kill thresholds must be CALIBRATED before the first trial**, and the data for it is
already on disk: four 10-epoch WS traces (`ws10`, `w10lf`, `cw10lf`, `lfwu20k`).

### DONE 2026-09-22 -- the calibrated rule

`scratchpad/wstune/` (`parse_val.py`, `calibrate_kill.py`, `kill_rule.py`, `kill_check.py`,
`smoke_kill.py`). The verdict tool is **`kill_check.py <candidate ws log> <reference ws log>`**;
exit 0 LIVE, 1 KILL, 2 PENDING.

> **KILL iff, at step 32,805 (epoch 3 of ten), the mean of the last 5 validation checkpoints is
> worse than the incumbent by >= 0.0010 in BOTH modes.** One decision, at that step, and nowhere
> earlier.

Three things the calibration settled, each of which changed the design:

1. **The inherited deltas are vacuous here, by a factor of a hundred.** The 0.004 / 0.006 pair was
   set where the curves fall ~0.01 per 1000 steps. Measured on these four traces, the fall at
   epoch 3 is **0.0001 per 1000 steps** (and 0.00003 by epoch 5). A 0.004 bar would spare every
   trial including a disaster.
2. **A single checkpoint cannot carry the decision.** The NULL pair (`cw10lf` vs `w10lf`, the same
   recipe on different hardware) reaches |d| = 0.0017 ahead / 0.0011 imm on individual
   checkpoints, which OVERLAPS the near-tie arm's range -- so any single-checkpoint threshold
   either kills the null or spares everything. Averaging 5 checkpoints collapses it: over **107
   sliding windows the null pair's both-modes-worse floor never exceeds +0.00012**, against a
   0.0010 threshold, for a measured false-kill rate of **0 of 107 at every threshold from 0.0005
   up**. The both-modes conjunction does most of that work -- the null "candidate" is slightly
   BETTER on imm throughout.
3. **"Fire at any checkpoint up to epoch 3" is REFUTED, and this is the load-bearing part.**
   `lfwu20k` fires at step 1000 with a **+0.1492 ahead** gap, at every threshold tested, because
   its one knob IS the warmup (400 -> 20,000) so its LR is still ramping. By epoch 3 the gap is
   +0.0008 / +0.0011 and by epoch 10 the ahead gap has **flipped sign to -0.0006**. Knob 4 of the
   grid below is warmup: an any-checkpoint rule would kill, by construction, the knob it is most
   needed for. That is the 2026-07-08 late-bloomer failure reproduced at 10-epoch scale.

**0.0010 rather than the 0.0005 the null alone would allow**, precisely because `lfwu20k` sits at
+0.0008 / +0.0011 at epoch 3 and then crosses over. D = 0.0005 kills it; D = 0.0010 spares it. An
arm that crosses after the decision point is the one failure this rule must not have -- and it
costs pruning rate, which is the trade taken deliberately.

**Non-vacuity:** `smoke_kill.py` builds synthetic candidates by shifting a real log's validation
means and requires KILL at +0.0015 in both modes, LIVE at +0.0030 in ONE mode (the conjunction is
load-bearing), LIVE at +0.0009 (the threshold is where it says), and PENDING on a truncated log.
All five pass.

**Limits, stated because they bound what this can promise:** n = 1 null pair, and it is
cross-HARDWARE; and the calibration traces are 561k runs while the trials are 332k. Each trial is
paired against its OWN 332k reference at the same steps, so a scale shift cancels in the
difference -- but the VARIANCE of a 10-user validation at 332k is unmeasured. Re-read the first
trial's window spread against these numbers before trusting the rate.

## Cost, measured

| | local | box |
|---|---|---|
| 10-epoch WS, 109,350 steps @ 0.976 steps/s | 31.1 h | 15.0 h |
| killed at epoch 3 | 9.3 h | 4.5 h |
| decay + full eval (only for an adoption) | 8.9 h | 3.6 h |

At 50% pruning the average trial is **20.2 h local / 9.7 h box**.

## The knobs and the order (my call)

| # | knob | env | grid | why here |
|---|---|---|---|---|
| 1 | LR scale, AdamW **and** Muon together | `PEAK_LR` (arg 9) x `RWKV_MUON_LR`, same factor | 0.5x, 2x | the largest lever in every earlier tune (iter 34 was the phase's biggest gain) and never tuned at 10 epochs or at 332k. Everything else is measured against whatever this settles on, so it cannot go second. ⚠ `PEAK_LR` alone moves ~10% of the weights -- Muon has its own base LR and governs ~500k of them |
| 2 | Muon-vs-AdamW **ratio** | `RWKV_MUON_LR` alone | 0.5x, 2x | the largest UNTRIED knob. Every previous tune moved the two together or moved AdamW alone, so the ratio has never been a variable |
| 3 | `MAX_TRAIN_GLOBAL_LEN` | 65536 -> 98304 | 1 trial | +14.3% rev/s but **-34% optimizer steps**; that trade was priced at 1.25 epochs and the step budget is what changed most at 10. Must come after 1, because the right LR depends on how many steps there are |
| 4 | warmup | `WARMUP_STEPS` 400 -> 2000 | 1 trial | 87 `lfwu20k` rejected 20,000 at 561k (+0.000236 ahead at p 0.014); the middle is untested and a smaller model may want a different ramp |
| 5 | epochs | 10 -> 15 | a CONTINUATION, not a trial | asks whether the budget itself is right at 332k. **It is far cheaper than it looks:** the WS schedule is warmup then CONSTANT LR, so `EPOCHS` sets only how many steps are taken and a 15-epoch WS IS the 10-epoch WS continued. It costs the 5 extra epochs as a resume from the incumbent's WS-final (~15.5 h local / ~7.5 h box), not a 46 h retrain -- and the kill rule does not apply, because through step 32,805 it is the reference |
| -- | wd, dropout, SAM | -- | **CLOSED** | the ws10 screen says the model is FIT-limited at 10 epochs, and a 1.69x smaller model is more so, not less. Four independent readings agree (tuner cut dropout, wd 0.2 lost, `lorawd` tied, SAM regressed) |

Round 1 = 7 trials. Round 2 re-tests the winner's neighbours where a knob won at a grid edge, ~4
trials. **~11 trials**, which the budget below covers.

## The trial machinery (built 2026-09-22)

`scratchpad/wstune/mk_ws_trial.py <incumbent runner .cmd> <knob> <value>` writes the Windows runner,
the WS toml and the cloud runner for one trial. Round 1 is generated and PREFLIGHT_ALL_PASS:
`wtlr0p5`, `wtlr2`, `wtmuon0p5`, `wtmuon2`, `wtwarmup2000`.

**A trial is WS-ONLY** (`RWKV_WS_ONLY=1`): no decay, no eval. It is ranked by its 10-user validation
endpoint read beside the incumbent's at the same steps, which costs no GPU. A decay + full eval is
bought only for a trial about to be ADOPTED -- roughly a third off every survivor.

**The kill rule rides in the trainer**, not in a watcher: `RWKV_VPRUNE_WINDOW_MEAN=5`,
`MIN_STEP == MAX_STEP == 32000`, deltas 0.0010 both modes, persist 1, against
`scratchpad/wstune/ref_<incumbent>.json` (built by `mk_vprune_ref.py` from the incumbent's WS log).
A pruned WS exits 42, and the cloud runners now treat that as a RESULT: no retry, no decay, no eval,
a `WS_PRUNED` line and `DONE_EXIT_0`, so the supervisor walks on to the next tag.

**Every override is asserted twice** -- on the output runner, and at RUN time against the trainer's
own echo of the value (`muon_lr=<v>`, `Warmup steps: <n>`), each with its own exit code. The
recurring tuner failure is a lever that silently does nothing and measures as a perfect null.
⚠ `mk_cloud_runner.py` builds the `.sh` from a TEMPLATE, so it now CARRIES ACROSS any WS-log guard
the `.cmd` declares; before that fix a guard added to the Windows runner did not exist on the box,
which is where the trials run. Verified inert: the three queued runners regenerate byte-identically.

⚠ **The `max` knob needs a measured group count.** Groups per epoch is a property of the train db at
that MAX (10,935 at 65536, unknown at 98304), and a wrong `WSSTEPS` silently breaks the resume
loop's completion test -- so `mk_ws_trial.py` refuses `max` without `--groups N` rather than
guessing. **MEASURED 2026-09-22: 7,208 groups per epoch at 98304** (`scratchpad/wstune/count_groups.py`; seed-independent; 65536 reproduces 10,935), i.e. -34.1% optimizer steps. `wtmax98304` is generated with WSSTEPS 72,080 and a kill step of 21,000 (epoch 3 in its own count). ⚠ The val-prune pairs by exact STEP, so at step 21,000 this trial has seen ~1.5x the reference's data: the rule is biased toward SPARING it -- the safe direction. Its run-time guard is the trainer's `Number of groups: 7208` line, because the trainer never prints MAX; a guard on a string it never writes would have failed after a full WS.

## Budget (Andrew 2026-09-22: **$57 at vast.ai, +$20 available, that is the ceiling**)

$57 = ~116 h of box at $0.493/h; $77 = ~156 h. Already committed: `selffeed` tail, `rowfeat`,
`s64hc10`, `r2s64h` ~ 34 h. So ~82 h of box remain at $57.

With 10 days of local GPU and ~82 h of box, at 50% pruning: **~12 local + ~8 box = ~20 trials** --
comfortably more than the 11 the grid needs, which leaves room for the second round and for the
extensions if a knob wins at an edge.

## Order of everything, and why R2 comes first

1. **`r2s64h`** -- the monotone-by-construction head on the accepted architecture (Andrew: *"don't
   forget to also run monotone-by-construction curve before all this"*). It goes first because if it
   is ADOPTED it removes the four PAVA probe rows per scored review from training AND eval, so every
   later trial is cheaper and needs less VRAM. Deciding it afterwards would mean tuning the wrong
   recipe and paying the probe cost throughout. ⚠ 97 `r2s80` already rejected it at d=80
   (-0.000378 ahead / +0.000249 imm); this asks on the architecture that ships.
2. **`s64hc10`** -- the 10+1 confirmation that gates the whole thing, and it doubles as the incumbent
   WS trace the tuning ranks against.
3. **Calibrate the epoch-3 kill rule** from the four existing traces (CPU, free).
4. **WS tuning**, knobs in the order above.
5. **Phase 5** (the QAT decay) on the tuned WS -- in that order, because tuning a decay onto a WS you
   are about to change wastes the decay work.
