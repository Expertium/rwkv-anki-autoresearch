# ordcut -- pre-registration (written 2026-09-04 19:40, before launch; ADOPTED slot)

**Lever:** `RWKV_ORD_LAMBDA=0.25` -- ordinal ONE-cut supervision of the forgetting curve's logit from
the NEXT review's rating, a label every real row carries (`label_rating`, `data_processing`'s per-card
shift) and nothing consumes today (`p_loss` is masked to query rows). Term, on successful ahead rows
with t >= 1 d: BCE(z - cut, target = next rating >= Good), cut = a + c*log1p(t / 1 d), (a, c) two
train-only conditional Parameters, zero-init (=> at start the term is the plain BCE on "not Hard").
Deploy unchanged: the served quantity is still sigmoid(z).
**Provenance (adopted):** Cao, Mirjalili & Raschka 2020, CORAL (arXiv 1901.07884), and Frank & Hall
2001 (ordinal classification as K-1 binary tasks on a shared score); proposed independently by two
of the three 2026-09-04 agents. Reduced from two cuts to ONE by the screen (below).
**Control:** whichever reference the lineage has when it launches -- realcyc, or durdrop if durdrop
promotes (the waiter's `auto_control.py` applies the mechanical gate and regenerates the runner).
Single-variable: the flag is the only diff (both-direction generator guards; params guard 563,654 =
reference + 2).
**Gate:** CURVE-SIDE exception -- the rating head and its objective are untouched, imm can move only
through the shared trunk: ahead raw >= +0.0001 at p < 1e-4, AND imm not significantly worse
(`paired_pvalue.py --curve-side`). Size gate 0/2499.

**The screen that shaped it (`proposals_2026-09-04/screen_pass.py`, realcyc, 10 train users,
125,236 successes at t >= 1 d):** the Hard share falls perfectly monotonically with the model's own
logit R (0.166 -> 0.016 across deciles, Spearman -1.000); the Easy share is U-shaped (0.143 -> 0.053
-> 0.099). AUC(Good vs Hard) on logit R = 0.737 -- separated but not solved (kill line 0.75). So a
shared latent explains Again < Hard < Good and NOT Easy; a second cut would distort calibration.

## Predictions

- **P1 (direction, ahead).** Rectified ahead improves: band **+0.0001 .. +0.0004**. Mechanism:
  target-variance reduction from a label the row already owns (KD's channel, 4/4 accepts, but with
  no teacher) -- a Hard success says "R was barely above threshold", which the binary label cannot.
- **P2 (imm).** Inside the floor (|delta| <= 7.5e-5); harm test not significant.
- **P3 (engagement).** `ord_cut_a` moves off 0 by >= 0.3 logits and the ordinal term's average falls
  below its a=c=0 value (readable from the ord params in the checkpoint + a re-run of the screen on
  the candidate: AUC(Good vs Hard) on the candidate's own R must RISE above 0.737). If (a, c) stay
  at ~0, the term was inert and the verdict is uninterpretable.
- **P4 (falsifier).** Ahead inside the floor WITH P3 holding = the curve's own logit already encodes
  the Hard/Good distinction as well as a shared cut can, and the label carries no extra shape
  information at this budget. Then close the sub-family (do not try the Easy cut; the screen says
  it is wrong-shaped).
- **Abort line.** Ahead worse by > 0.0002 (the cut fights the Again boundary; retry once with
  `RWKV_ORD_MIN_T=259200` and lambda 0.1 before closing).

## Not redundant with (stated before the number)
iter 46 (soft targets from the imm HEAD -- a model output, not a label); iter 48 (R(t) into the rating
logits -- routing); iters 17/19 (pbin: the ahead label put on the RATING head); iter 11 (grade
EMBEDDING on the input side). This adds no path between heads; it reads a label.
