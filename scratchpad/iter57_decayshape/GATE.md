# decayshape (`RWKV_DECAY_SHAPE=linear`) — annotations for a RUNNING experiment

Written 2026-08-19 while the run is live. **Annotations go in a separate file, never in the runner:**
cmd.exe re-reads a batch file from a saved byte offset after every command returns, so editing a
running `.cmd` makes it resume mid-garbage. Iters 43 and 46 died that way.

## The run is CORRECT. Verified in its own decay log, not read off the runner

```
[kd-mix] KD ON: ... alpha FIXED at 0.5
[decay-shape] LR decay shape = linear (default is 1-sin(pi x/2), mass 0.3634)
```

* env sets `RWKV_KD_ALPHA=0.5` (line 103) and `RWKV_DECAY_SHAPE=linear` (line 105);
* the guard checks `alpha FIXED at 0.5` (line 132) — this is the guard that was **wrong** 90 seconds
  into the first launch (it checked 0.9) and was fixed before relaunch, at a cost of ~1 min of GPU;
* decay checkpoints land in `scratchpad/iter45_kddecay/` (`i57_d_*.pth`), not here — a decay-only run
  writes beside the WS checkpoint it decays from. An empty run directory is expected, not a failure.

## One cosmetic defect, found by the new preflight check. NOT worth touching a running file

Two log lines carry iter 52's prose:

| line | text | reality |
|---|---|---|
| 85 | `echo ===== ITER 57 (KD alpha_decay 0.5 to 0.9) START` | the lever is `RWKV_DECAY_SHAPE=linear`; alpha is 0.5 in both this run and its baseline |
| 143 | `echo ITER57 DECAY_OK KD ON alpha 0.9` | alpha is **0.5** — will be written when the decay finishes |

Both are `echo`, so nothing computed depends on them. **The run was never at risk; the record is.**
The log is what a verdict gets read from months later, and the number this run produces will sit
beside a log line announcing an alpha it never used.

Same defect as `kdalpha025`, which announced `alpha 0.9` for a run whose alpha is 0.25 — that one was
still queued, so it was fixed in its generator (three prose lines, no executed logic touched).

`preflight_runner.py` now reports this class as a NOTE rather than a failure: an echoed line that
names an alpha the runner never sets. It is a note and not an error because a `REM` may legitimately
discuss the value being moved away from, and the check has to distinguish prose from logic.

## Pre-registered gate rule — written 04:25, BEFORE any number was read

The eval was at 2202/2500 users and no result had been inspected when this was written. Fixing the
rule now is the point: picking `--curve-side` *after* seeing that imm regressed would be gate
shopping, and the curve-side exception is narrow enough to be tempting.

**This lever gets the BOTH-MODES rule, not the curve-side exception.** `RWKV_DECAY_SHAPE=linear`
changes the LR schedule of the decay phase, so it acts on the **whole trunk** through the optimizer.
CLAUDE.md scopes the exception explicitly: it is for levers that touch only the curve/ahead objective
(self-distillation, PAVA lambda, ahead-target and monotonicity changes), and states that
"trunk / optimizer / capacity / topology changes keep the BOTH-modes rule, because those genuinely
can move both."

So the command is the plain form — **no `--curve-side`**:

```
paired_pvalue.py --cand-ahead RWKV-iter57_decayshape --cand-imm RWKV-P-iter57_decayshape \
                 --champ-ahead RWKV-iter53_muonlora --champ-imm RWKV-P-iter53_muonlora --intersect
```

Accept only if **raw ≥ 0.0001 in BOTH modes** vs iter 53 **and** p < 0.0001 in both.

**Prediction: null.** iters 41/43/44 established that three structurally different arrangements at
identical capacity are mutually indistinguishable at |delta| ≤ 7.5e-5, and a decay-shape change is a
reallocation of the same optimization budget rather than more of it. The honest prior is that the
WSD decay tail is already near-flat in outcome, so reshaping it moves little. A null here bounds the
schedule-shape family and is worth recording as such.

## When recording the verdict

* Gate against the **current champion, iter 53** = `0.297523 ahead / 0.265191 imm` (VAL 5001–7500).
* This run was built on the **iter-45** recipe, so report **both** deltas: vs iter 45 is the
  controlled effect of the lever, vs iter 53 is the gate.
* Ignore the "ITER 57" label in these log lines — the number is assigned at verdict time under the
  completion-order convention, and the directory digits do not bind it. `exp` in
  `research_log.jsonl` is the identity.
