# CORRECTION to the 2026-09-06 round: the eval does NOT chunk, so the cold-start finding is a TRAIN-ONLY mismatch

Written 2026-09-07 07:00, after measuring the chunk lists in both databases directly. It corrects the
round document's rank 2 (learned initial state) and rank 4 (chunk-continuous training), and it
supplies the number a SHELVED plan has been missing since June.

## What I claimed, and what is actually true

The round's addendum said the reset cost "could be ~0.001+ ahead on multi-chunk users -- but only if
EVAL carries state too, which changes how the metric is computed". That framing assumed the eval
chunks a user's history the way training does. **It does not**, and the databases say so plainly:

| database | users | chunks/user (median) | multi-chunk users | rows within 2048 of an INTERIOR boundary |
|---|---|---|---|---|
| `train_db_5k_h1_id5` | 5,000 | **4** (max 313) | **88.3%** | **12.50% of all rows** |
| `test_db_5k_id5` | 2,500 | **1** (max 4) | **0.2%** | **0.00% of all rows** |

Median rows per user is ~51k in BOTH databases, so this is not a size difference -- the test builder
keeps a user whole and the train builder cuts at ~16,384. `STATEFUL_BPTT_PLAN.md` step 4 already said
so in one line ("Eval already carries implicitly (full history, 1 batch/user)"); I measured the reset
cost without reading it.

**Consequences.**
1. **A learned initial state cannot recover metric loss at chunk boundaries, because the metric has
   almost none.** 3 of 2,499 VAL users have an interior boundary at all. Rank 2's projected by-user
   gain via that route is **+0.000000**, not +0.0000..+0.0003.
2. **The cold-start cost I measured (+0.024..+0.028 ahead on a chunk's first 256 rows, +0.0043 /
   +0.0118 mean over the first 2,048) is real but lands entirely on TRAINING.**
3. So the finding is not "a recoverable metric loss". It is a **train/eval structural mismatch**:
   **12.5% of training rows are computed from a cold-started state that the eval and the deploy
   engine never impose**, at a measured handicap of ~+0.004..+0.012 ahead BCE in that window.

## The mismatch is bigger than the cold start alone

Training only ever sees contexts of <= ~16,384 rows. The eval scores users whose median history is
**51,048 rows** (max 7.7 M) in ONE sequence. So the model is fit on short contexts and scored on long
ones -- which is also why `RWKV_STATE_CLAMP_TAU=300 / WINDOW=32768` exists. Cold starts are one
symptom of that; the other is that no training gradient has ever seen a state older than ~16k rows.

## What this changes in the queue

* **Rank 2 (learned initial state): DEMOTED, and re-motivated.** Its metric route is dead. What
  remains is a training-conditioning argument -- a shared learned init is a better starting point
  than zeros for the 12.5% of rows that start cold -- plus a small eval-side effect at each user's
  review 0 (mostly unscored: the first sixth of a history is never scored). It also carries real
  deploy debt (model-forward change -> §9 three-way parity + a fresh trace + a Rust port), which the
  round under-costed as "1 run + trace".
* **Rank 4 (chunk-continuous carry): PROMOTED on evidence, still Andrew's call on the days.** It is
  the only lever that removes the mismatch rather than softening it, and it now has a measured target
  (12.5% of rows, +0.004..+0.012 each) instead of the round's guess.
  **★ AND MOST OF IT IS ALREADY BUILT.** `optimization/STATEFUL_BPTT_PLAN.md` records the CUDA
  stateful kernel as DONE and parity-verified: `stateful(state0=0)` equals the non-stateful op at
  **exactly 0**, forward split-equivalence `fwd([A;B]) == [fwd(A); fwd(B, state0=final_A)]` at
  **exactly 0**, truncated-BPTT grads at 3.8e-6. What remains is steps 1-3 (model-forward carry, the
  per-entity state store, synchronized stateful batching), described there as intricate and
  multi-day. **The plan was shelved on SPEED** ("smaller chunks don't speed training") -- the
  accuracy case was listed as a benefit and never quantified. It now is.
* **A CHEAP TEST OF THE WHOLE HYPOTHESIS EXISTS and needs no kernel work:** down-weight the rows in
  the post-boundary recovery window in the loss, exactly the instrument `RWKV_EQUALIZE_LOSS_W` uses
  for unscored rows. If those 12.5% of rows are computed on degraded states, their gradients are
  partly about a condition that never occurs at scoring time. ~5 lines, train-only, zero deploy debt.
  ⚠ Sequence it AFTER eqw: it is the same mechanism (row reweighting) and eqw's verdict is evidence
  about whether this family pays at all.

## The methodological lesson

I measured a real effect (the cold-start cost) with a correct instrument, and attached it to the
wrong population. **Before sizing a lever, check WHICH ROWS THE METRIC ACTUALLY SCORES** -- the same
check that produced eqw one day earlier, applied one level up: not "which rows are scored within a
sequence" but "how the scored sequences are cut". Both databases' chunk lists are two lines of LMDB
reads; the round's estimate cost nothing to verify and I verified it only after writing it down.
