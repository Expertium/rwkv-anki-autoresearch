# Pre-registration: how much of the -id lineage's accuracy is the query-row clock leak? (2026-09-10 20:30)

## The claim (Discord, relayed by Andrew 2026-09-10)
`t_since_any_review` (id_features.py:571) and the sibling gap (:439) are measured to the reconstructed
SHOW time `id - taken_millis`, and survive query masking (data_processing.py:540 keeps every new
column). "A gap of a few seconds between previous_review_end and current_review_start almost never
indicates a real gap. In normal Anki usage, it indicates some information about how the review was
treated after it was first shown."

## Confirmed in the raw -id data BEFORE any model run (`mechanism.py`, users 5001-5040, 2.36 M reviews)
| gap before the card | share | P(fail) | P(capped) | P(fail), uncapped only |
|---|---|---|---|---|
| 0-1 s | 71.4% | 15.1% | 0.08% | 15.1% |
| 2 s - 30 min | 11.2% | 21-30% | 12-36% | 20-28% |
| 30 min - 3 h | 0.45% | 25.8% | 23.6% | 24.3% |
| > 3 h | 0.92% | 21.4% | 10.9% | 20.5% |
Two channels: (1) the duration CAP (60 s default; 30 of 40 users) -- a capped review's show time is late
by its excess, median in-session gap 47 s vs 0.012 s uncapped; (2) uncapped reviews with a 2 s - 30 min
gap still fail at 20-28%, so something else delays the reconstructed show time (a timer reset on edit or
leave-and-return is the likely one). The gap bucket alone carries 0.009 nats about the outcome.

## What a deploy scheduler sees, and why only query and probe rows leak
The next card appears the moment the previous one is answered, so at prediction time the gap is ~0 in
a session. QUERY rows (the imm prediction) and PAVA PROBE rows (the button intervals shown BEFORE the
answer, i.e. the rectified ahead metric) are predictions made at show time and copy the current
review's clock columns -- they leak. REAL rows describe finished reviews, and deploy rebuilds them from
the same revlog fields, so they are legitimate. The ahead LABEL time (the next review's own excess) has a
smaller version of the same channel; it is NOT covered here.

## The measurement -- eval-only, no production file edited
`get_result_cf.py` patches `insert_probes` (fetch workers see it: proven by a spawn probe). On query and
probe rows, delta = the in-session gap (< T); every column measured to the show time moves back by delta
("the card appeared at the previous answer"): t_since_any_review -> 0, sibling gap and same-card
interval (+ cumulative) minus delta, the daily sin/cos pairs rotated back. Unit-tested on real gen-5
chunks: T=0 changes nothing; T=1800 changes only in-session query/probe rows, never a real row.
Arms on arm 1's checkpoint `w10p_d_21870`, users 5001-5300, rectified metric: lkc0 (T=0, control),
lkc30m (T=1800), lkc3h (T=10800). Runs at the START of arm 2's eval runner, ~1 h, never fatal.
Harness check: lkc0 must reproduce arm 1's own full-run numbers on these users.

## What the number means
lkc30m - lkc0 is how much the eval overstates THIS model at deploy-faithful inputs. A model retrained
without the leak can recover part of it from legitimate inputs, so it is an upper bound on what the fix
costs the best leak-free model, and a lower bound on nothing.

## Predictions
* imm, 30 min: **+0.002 .. +0.005** (LOO zeroing t_since_any_review on ALL rows cost +0.005076; the
  features' whole imm gain was -0.0024 while ahead gained -0.0003).
* imm, 3 h >= 30 min (it also removes the 30 min - 3 h rows, which are still 24% capped).
* rectified ahead, 30 min: **+0.0002 .. +0.0008** (probe rows only; real rows keep the gap).

## Decision rule (imm at 30 min)
* **>= +0.0010 MATERIAL.** (1) The -id lineage's imm numbers stop being deploy numbers: "imm stop
  criterion met" and "beats the d=128 model on imm" are withdrawn until a leak-free model is measured.
  (2) Build the production fix = this same transform, env-gated, in train, eval and deploy (Python RNN +
  Rust), with a three-way parity case. (3) Queue a leak-free decay branch from ws10 as the new reference.
  The relative A/Bs in the chain (QAT tax, KD, capacity) share the leak in both arms and keep running.
* **+0.0003 .. +0.0010 SECONDARY.** Fix with the next rebuild; re-read the stop criterion with the
  number subtracted.
* **< +0.0003 MINOR.** The data carries the leak but the model barely uses it on query rows.

## Note added 2026-09-10 22:10, BEFORE phase L has run: the decision is now applied mechanically

The production fix exists (`RWKV_CLOCK_AT_PREV_ANSWER`, `rwkv/clock_fix.py`, smoke
`smoke_clock_fix.py` PASS) and the leak-free branch is built (`scratchpad/w10clk/`). The rule above
is applied by `scratchpad/w10clk/clk_decide.py`, called by `wait_qateval_then_clk.cmd` once arm 2's
eval runner (which holds this phase at its start) ends `DONE_EXIT_0`: MATERIAL => the branch runs
next, at T = 1800 unless lkc3h costs >= +0.0005 imm more than lkc30m (then T = 10800); below
MATERIAL, or a harness that does not reproduce arm 1 within 0.0003, => the branch is not inserted
and the queue continues. The rule's thresholds are the ones written above; nothing was moved.
