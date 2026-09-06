# eqw -- pre-registration (written 2026-09-06 05:05, before launch; INVENTED slot after muonscale)

**Lever:** `RWKV_EQUALIZE_LOSS_W=0.25` -- "train on what is scored". The benchmark scores only the rows
flagged `label_is_equalize` (each user's TimeSeriesSplit(5) test folds after the delta_t > 0 rule), so
the first sixth of every history is never scored, and on realcyc's 218,841 train-user rows those
never-scored rows are a different, EASIER distribution: by-user ahead BCE **0.189 vs 0.279**, failure
rate 7.0% vs 13-15%. Training weights every labelled row equally, so ~17% of the fit is spent on rows
the metric never sees, in a model the iter-65 result showed to be FIT-LIMITED. The flag weights
unscored rows by 0.25 (scored rows 1) in the ahead AND imm objectives -- weighted means, so the loss
magnitude and the tuned LRs hold; the equalize metrics stay unweighted. Train-only, no parameters, zero
deploy debt; default 1.0 = byte-identical. Precedent in the repo: the PAVA probes are already restricted
to scored rows (`probe_equalize_only=True`).
**Control:** the reference at fire time -- muonscale if it passes its both-modes gate, else muonscale's
own control (hord if hord passed, else realcyc). `auto_control.py` decides mechanically; `mk_eqw.py <base>`
regenerates the runner on that base so the flag is the only difference.
**Gate:** BOTH-modes rule (both metrics score the same rows, and the trunk is shared): ahead AND imm raw
>= +0.0001 at p < 1e-4; size 0/2499.

## Predictions
- **P1 (direction).** Both modes improve: band ahead **+0.0000 .. +0.0003**, imm **+0.0000 .. +0.0003**.
  A null (both inside the +/-7.5e-5 floor) is the counter-hypothesis: the early rows are not "wasted"
  fit -- they are cheap supervision on the same function, and where the fit is spent does not matter
  at this budget. That would close the where-the-fit-is-spent axis with a mechanism.
- **P2 (engagement, recorded BEFORE the number).** The WS step trace's reported ahead objective is a
  weighted mean and must sit ABOVE realcyc's at matched steps (the unscored rows are the easier ones);
  the `ahead_equalize_avg` metric (unweighted, scored rows only) is the comparable quantity. On the
  candidate, the by-user ahead BCE on the unscored first sixth of train users should RISE relative to
  realcyc's 0.189 (the fit moved away from them) while the scored rest falls; if the first-sixth loss
  does not move, the weighting did not reach the optimizer and the verdict is uninterpretable.
- **P3 (the informative failure).** A REGRESSION beyond -0.0001 in either mode means the early rows
  carry supervision the later rows need (the model learns how histories START from them: the
  state-dynamics view), and that down-weighting them starves it. Then the where-the-fit-is-spent axis
  is closed in the OTHER direction and no milder alpha is worth a run (it only interpolates).
- **Abort line.** Either mode worse by > 0.0002.
- **Retry.** None registered. alpha 0 (hard mask) is NOT a retry: it makes a user's first chunk a
  zero-gradient sequence and is dominated by 0.25 on both outcomes.

## What it is not
Not iter 37 (by-USER weighting to match the by-user mean -- refuted in every size quartile): this
selects ROWS to match the scored set at unchanged per-user weight. Not a regulariser (the
fit-limited finding says those cannot pay): it moves fit between rows, adding none and removing none.
