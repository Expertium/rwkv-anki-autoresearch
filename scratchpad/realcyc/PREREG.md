# realcyc -- pre-registration (written 2026-09-04 01:50, eval running, NO number seen)

**Lever:** `RWKV_REAL_CYCLES=1` on the gen-5 dbs. The 7 pseudo day-offset cycles (28 encoding dims,
seeded phase from the user's first day) and pseudo `day_of_week` (row 11) are replaced by 24 real
UTC-anchored cycle columns (3 d / week / month / year / decade / century, sin+cos, review-time and
first-review halves). Input 114 -> 109, params 565,252 -> 563,652.
**Control:** gen4base (0.298089 / 0.263548, n=2,499, user 6701 excluded). Same recipe, same label
filter (`label_filter_db_id_e2s`), KD off in both, seed 4321. Single-variable: the flag + the db
generation it requires (gen 5 = gen 4 + the cycle columns; entry counts and equalize sets verified
IDENTICAL at build time, `EQUALIZE_MATCHES_GEN4`).
**Gate:** BOTH-modes rule -- raw >= +0.0001 in each mode AND paired one-sided Wilcoxon p < 1e-4 in
each (`paired_pvalue.py --intersect`); size gate = 0/2499 vs `size_baseline_id_e2s.json`.
**Caveat carried:** the WS was resumed at step 3,000 after the SSD move (dropout draws of the tail
differ; weights/optimizer exact). Statistically equivalent to an uninterrupted run, not bit-identical.

## Predictions

- **P1 (direction).** Both modes improve, ahead by more than imm in RELATIVE terms. Mechanism: the
  ablation showed the pseudo cycles are dead weight (+0.000083 / -0.000005 reliance), so removing
  them costs nothing; the bet is that REAL calendar phase carries information the pseudo phase could
  not (a shared phase across users lets the model pool weekly/annual patterns across the population).
  Band: ahead +0.0001 .. +0.0006, imm +0.0000 .. +0.0004. imm already sees `time-of-day`/`dow`/`doy`
  of the target review through the query row's clock columns (the ablation's clock group), so the
  weekly and annual halves are partly redundant for imm; ahead has no target-row clock at all.
- **P2 (where).** The gain concentrates in users with LONG histories: annual/decade/century phases
  need > 1 year of data to be learnable within a user, and the shared anchoring pays most where the
  user's own history spans several cycles. Test: Spearman rho between per-user delta (ahead) and the
  user's history span in days, expected > +0.10; top-vs-bottom span quartile ratio > 1.5x.
- **P3 (null shape, if P1 fails).** If both modes are inside the +/-7.5e-5 floor, the reading is
  "calendar phase carries no information beyond what the clock columns already give" -- and the
  drop of 28 pseudo dims + row 11 for zero cost is still an accepted SIMPLIFICATION on the size/speed
  exception (5 fewer input dims, 1,600 fewer params), not an accuracy accept.
- **Abort line.** Either mode WORSE by more than 0.0002 => the real phases are being used badly
  (e.g. century/decade columns are near-constant within the dataset's span and act as a per-user
  identifier), and the next variant drops the decade/century pairs before anything else.

## What would change the plan

- P1 holds and gate passes: realcyc promotes; `mk_lorawd.py realcyc` BEFORE lorawd's waiter fires
  (it polls `wait_realcyc3.log` for an anchored `DONE_EXIT_`, written by the resume wrapper).
- P1 fails but P3 holds: gen 5 stays the lineage's db (simpler inputs, same accuracy); lorawd keeps
  its gen4base control; the LOO sweep decides which real cycle columns stay.
