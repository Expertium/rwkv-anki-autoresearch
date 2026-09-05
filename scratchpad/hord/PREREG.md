# hord -- pre-registration (written 2026-09-05 19:00, before launch; INVENTED slot after sam)

**Lever:** `RWKV_PAVA_HORIZON_LAMBDA=0.05` (factors 0.125 and 8): on the probe rows only, a hinge
`relu(R_b(t_h) - R_{b+1}(t_h))` over the three adjacent-button junctions of the 4 counterfactual
curves at horizons t_h = t_label x {1/8, 8} (clamped to [10 min, 1 y]), evaluated on the GRU curve
parameters of the probe rows, no label at t_h. A pure ORDERING regulariser: PAVA rectifies the probes
at ONE t (the target's); the scheduler chooses t from the pressed button, so each counterfactual curve
is supervised only in its own button's t-range and nothing orders them elsewhere. Curve head only. No
new params (563,652). Deploy unchanged (the same rectifier and solver; the curves are trained to
need pooling less).
**Provenance:** invented (domain agent, 2026-09-04 refill rank 5; the constraint is FSRS's
S(Again) <= S(Hard) <= S(Good) <= S(Easy), which implies R_b(t) ordered in b at every t).
**Control:** the reference at fire time -- realcyc, or sam if sam passed the both-modes gate (then
hord's decay carries `RWKV_SAM_RHO=0.05` too, via `mk_hord.py realcyc --sam`, and the control is sam's
own numbers). `auto_control.py` decides mechanically.
**Gate:** CURVE-SIDE exception (ahead raw >= +0.0001 at p < 1e-4; imm not significantly worse); size
0/2499.

**The screen that motivated it (`button_probe.py`, realcyc, 1,478 probes over 4 train users):** raw
adjacent-button order violations on 29.9% of rows at the label's own t, 32.5% at 1 d, 32.1% at 7 d,
35.6% at 30 d, 48.8% at 180 d; the button ORDER differs from its label-t order on 11-34% of rows;
median |R_Good - R_Hard| at 30 d = 0.070. Smoke (`smoke_hord.py`): hinge 0.0108 on a real chunk at
init, exactly lambda x hinge added to the loss, the same-t PAVA term untouched, zero on coinciding
curves, gradient reaches the GRU head.

## Predictions
- **P1 (direction).** Rectified ahead improves: band **+0.0000 .. +0.0003**. This is a regulariser
  story, not a fix-an-error story: the claim is that coherence of the counterfactual family
  regularises the shared (w, S, d) readout, the same shape as PAVA lambda 0.1 -> 0.2's +0.00048.
- **P2 (imm).** Inside the floor; harm test not significant (probes never reach the rating head).
- **P3 (engagement).** Re-run `button_probe.py` on the candidate: the crossing rates at 1 d / 7 d /
  30 d / 180 d must fall by >= 50% vs realcyc's 32.5 / 32.1 / 35.6 / 48.8%. If they do not, the dose
  was too small and the verdict is uninterpretable; lambda 0.2 is the one retry. Also read
  `pava_pool_frac` at the label t from the trace: it must not rise.
- **P4 (falsifier).** Ahead inside the floor WITH P3 holding = ordering the counterfactuals off the
  label horizon carries no information about the pressed curve at the label horizon; then the
  curve-regulariser family stays at 2/4 and no third ordering constraint is proposed.
- **Abort line.** Ahead worse by > 0.00015 (the hinge fights the same-t pooling): one retry at
  lambda 0.02, then close.

## Not redundant with (stated before the number)
The killed spacing-effect lever (monotonicity in REVIEW COUNT; this is in BUTTON at fixed t, PAVA's
own axis); monotone-in-t (given by construction); lambda sweeps (same-t weight); the calendar-aware
curve (killed; unrelated); iter 64's ordinal cut (a LABEL on the curve logit at the label t; this
has no label and touches only off-label horizons).
