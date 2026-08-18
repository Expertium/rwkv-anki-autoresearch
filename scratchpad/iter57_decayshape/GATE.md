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

## When recording the verdict

* Gate against the **current champion, iter 53** = `0.297523 ahead / 0.265191 imm` (VAL 5001–7500).
* This run was built on the **iter-45** recipe, so report **both** deltas: vs iter 45 is the
  controlled effect of the lever, vs iter 53 is the gate.
* Ignore the "ITER 57" label in these log lines — the number is assigned at verdict time under the
  completion-order convention, and the directory digits do not bind it. `exp` in
  `research_log.jsonl` is the identity.
