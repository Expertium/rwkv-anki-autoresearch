# durdrop -- pre-registration (written 2026-09-04 19:30, before launch; INVENTED slot)

**Lever:** `RWKV_DUR_DROP=0.25` -- train-only per-row Bernoulli zeroing of the `scaled_duration` input
(dim 8) on the gen-5 SSD dbs. Query and probe rows already carry 0.0 there, so only REAL rows change;
eval and deploy are untouched (`self.training` gate; deploy contract unchanged: zeroed current
duration + PAVA + no residual). Descends from iter 33 (three bundled changes, rejected) and iter 18
(permanent removal, the p=1 end of the dose curve); iter 33's own write-up prescribed exactly this
instrument and it was never run.
**Control:** realcyc (0.298083 / 0.263592, n=2,499; same recipe, gen-5 dbs, KD off, seed 4321).
Single-variable: the flag is the only diff in `run_durdrop.cmd` (both-direction generator guards).
**Gate:** BOTH-modes rule (an input-side change reaches the shared trunk): raw >= +0.0001 in each
mode AND paired one-sided Wilcoxon p < 1e-4 in each; size gate 0/2499.

**The binding constraint, measured before building (`proposals_2026-09-04/screen_pass.py`):** on
realcyc, replacing the current review's duration by 0 costs **+0.001388 ahead** by-user mean over 10
train users (8 of 10 positive; iter 31 measured +0.001451 on the published set). That is the quantity
the rectified metric scores and the curve is trained on it through only the PAVA probe term
(lambda 0.2 x density 0.08 ~ 1.6% of the ahead weight).

## Predictions

- **P1 (direction, ahead).** Rectified ahead improves: band **+0.00015 .. +0.0005**. Mechanism: the
  curve head learns, at full loss weight on ~25% of rows, the input distribution the metric scores.
- **P2 (imm).** imm moves little: band **-0.0001 .. +0.00005**. The rating head already reads
  duration=0 on every query row; the risk is the STATE losing duration on p of rows (iter 18 says
  duration is real imm signal). Harm line for imm: -0.0001 (would make the both-modes gate fail
  regardless of ahead).
- **P3 (engagement, checked on the candidate's decayed checkpoint with the same screen instrument,
  ~40 min CPU):** the duration-zeroing cost falls from +0.001388 to **under +0.0007** (the model
  stops depending on the current duration for the curve). If it does NOT fall, the dose was too
  small and the verdict is uninterpretable rather than a null; then p=0.5 is the one retry.
- **P4 (falsifier).** Rectified ahead inside the +/-7.5e-5 floor WITH P3 holding = the curve head
  cannot serve both input distributions at this width; then the family (18, 33, this) closes at 0/3
  and the deploy penalty is a fact of deploy, not a training target.
- **Abort line.** Either mode worse by > 0.0002.

## What the number does NOT measure
The 30% PAVA-pooling half of the deploy penalty (+0.000611 at iter 31) is untouched by this lever.
