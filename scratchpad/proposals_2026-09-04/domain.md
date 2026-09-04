# Domain-prior proposals -- 2026-09-04 (memory science / spaced repetition / Anki usage)

Agent prior: FSRS structure (S0 by first rating, stability/difficulty dynamics, dual-stability
short-term memory), rating semantics (Hard = weak success, Easy = strong success), the Anki
scheduler's actual use of the model (rectified 4-button curves -> intervals; 1-P(Again) at now).

Reference = realcyc (gen-5 lineage, KD-off): ahead 0.298083 / imm 0.263592, n=2,499. AHEAD is the
binding mode (stop criterion 0.2950). Every proposal below is train-time or probe-time only, needs
no rebuild and no new input, keeps per-card state and the stream hierarchy, and ships nothing new
to Rust unless stated.

Two data screens were RUN while writing this (parquet only, 20 stride users 125..4875, ~1.6 M
labelled rows), so proposals 1 and 5 already carry their first kill-check result:

| screen | result |
|---|---|
| Does the grade at k+1 predict the lapse at k+2 at MATCHED k+2 interval? | Yes, for t >= 3 d: Hard 12.6-14.6% / Good 9.3-10.3% / Easy 7.7-11.7%; per-user within-bin Hard-minus-Good gap +0.0096, positive in **14 of 18** users; Easy-minus-Good -0.0174, negative in 10 of 15. Inverted at t < 1 d (learning steps). Grade shares Again/Hard/Good/Easy = 13.3 / 15.0 / 54.3 / 17.4% |
| How much of a card's history does a 16,384-row train window show? | The card's PREVIOUS review is visible with P = 0.884; on average **76.7%** of a card's prior reviews are inside the window (per-user min 36%, max 98%); only 3.1% of labelled rows have the previous review > 16k rows back |
| Share of rows that are a card's FIRST review | **16.2%** (per-user 3-39%) |

---

## 1. Graded (ordinal) curve supervision from the NEXT review's rating  -- rank 1

**Family:** objective / label design (curve-side). **Provenance:** invented (the domain fact is
FSRS's own: `hard_penalty` / `easy_bonus` on stability, `fsrs_v7.py:next_stability`, i.e. the
rating is graded evidence about the memory state, not just pass/fail).

**Mechanism.** `srs_model.py::_get_loss` supervises the curve head with `label_y` only -- the
BINARY outcome of review k+1 -- while `label_rating` on real rows (the k+1 rating, written by
`data_processing.py:429-439` via `groupby(card).shift(-1)`) is consumed by NOTHING: `p_loss` is
masked to query rows (`immediate_mask = is_query * has_label`). So every ahead row throws away
half its label. Add a cumulative-link (proportional-odds) term on the SAME curve logit `z =
curve_logits`: `P(rating >= j | z) = sigmoid(z - theta_j)` with **theta_1 pinned at 0** (so
`P(y=1) = sigmoid(z)` and the existing BCE is untouched) and two learnable thresholds
`theta_2 < theta_3`. By the chain rule the ordinal NLL equals `BCE + [-log P(grade | success, z)]`,
so the new term is exactly the CONDITIONAL grade NLL among successes, added with weight
`RWKV_ORD_LAMBDA` (0.25 first) on `ahead_mask` rows. Default 0 = byte-identical. Because the
t < 1 d bins invert the ordering (learning steps: Hard 2.1% vs Good 4.2% lapse), let the two
thresholds drift with horizon: `theta_j(t) = a_j + b_j * log1p(t / 86400)`, 4 params total.
Optionally mask the term to `t >= 1 d`. Nothing ships: thresholds are train-only; inference
reads `sigmoid(z)` exactly as today.

**Why AHEAD specifically.** It is a pure curve-head lever: the rating head, the trunk's imm
objective and the PAVA probe path are untouched (probes have `has_label=0`). The gain mechanism
is TARGET-VARIANCE REDUCTION, the same channel KD paid through (4/4 accepts), but from a label
the row already owns rather than from a teacher -- and this lineage runs KD-OFF because no
teacher takes the 109-dim layout, so ~0.0019 of KD value is currently unclaimed. A Hard at k+1
says "R was lower than a Good would imply"; a binary label cannot say that. 32% of successes are
Hard or Easy, so the graded signal has mass.

**Cheap CPU screen (the data half is DONE, above; the model half ~45 min).** Extend
`scratchpad/spacing_screen/calibration_by.py::collect()` to record THIS row's rating (it stores the
previous press as `r`; add `row["rating"]`), run 3 train-range users through the deploy RNN, then:
(a) proportional-odds sanity -- among successes with t >= 1 d, bin by decile of `logit p`: the Hard
share must FALL and the Easy share RISE monotonically (Spearman |rho| over deciles >= 0.8 for both);
(b) ordinal prize -- fit the 2-threshold link on `logit p` held out BY USER and report the
conditional-grade NLL it explains. **Kill:** (a) fails (grade not a monotone function of the
model's own R -> a shared latent is the wrong model, the term would distort calibration), or the
data screen's Hard-vs-Good gap had been < 1.2x at matched t (it is ~1.4x, so this half passed).

**Pre-registered band.** ahead **+0.0001 .. +0.0004** at p < 1e-4; imm within the floor (curve-side
gate). P-diag: `theta_2, theta_3` must separate (theta_3 - theta_2 > 0.3 logits) and `b_j` must
carry the short-t inversion (b < 0) -- if thresholds collapse to each other the lever was inert.
Abort line: ahead worse by > 0.0002 (calibration distorted; then retry with the t >= 1 d mask and
lambda 0.1 before closing).

**Not redundant with:** iter 11 (grade EMBEDDING, input side), iters 17/19 (pbin, an imm-side
binary term), iter 46 (soft target from the imm HEAD -- a model output; this is a LABEL), iter 48
(R(t) into the rating logits). The closed "routing imm -> ahead" family moved MODEL information
between heads; this adds no path and reads only a label the ahead row already has.

---

## 2. Button ordering across horizons -- multi-t PAVA on the counterfactual probes  -- rank 2

**Family:** curve-shape regulariser (the family's two wins: PAVA, lambda 0.2). **Provenance:**
invented; the constraint is FSRS's `S(Again) <= S(Hard) <= S(Good) <= S(Easy)` (clipper line
`w[1] >= w[0]` etc., and the monotone S-updates), which implies `R_b(t)` ordered in `b` at EVERY t.

**Mechanism.** `_pava_probe_loss` rectifies the 4 probe curves at ONE t -- the target's
`label_elapsed_seconds` (probes copy it, `prepare_batch.py:insert_probes`). The domain problem:
the scheduler chooses t from the PRESSED button, so each counterfactual curve is supervised only in
its own button's t-range (Again ~10 min .. Easy ~months); at the pressed t, three of four curves are
extrapolating and nothing orders them there. The Rust interval solver inverts the RECTIFIED curve
pointwise, so a crossing is not a deploy bug -- it is a pooled composite the trunk was never asked
to make coherent. Add, on the probe rows only, the curve at K=2 extra horizons `t_h = t * {1/8, 8}`
(clamped to [10 min, 1 y]) via `gru_forgetting_curve(out_w, out_s_raw, out_d_raw, t_h)` (all three
tensors are already in `_get_loss`), and penalise adjacent-button ORDER violations with a hinge
`relu(R_{b}(t_h) - R_{b+1}(t_h))` summed over the 3 junctions, weight `RWKV_PAVA_HORIZON_LAMBDA`
(0.05). No label at t_h -> a pure ordering regulariser; default 0 = byte-identical. Deploy
unchanged (the rectifier and solver are the same; only the curves are trained to need it less).

**Why AHEAD.** Probes and PAVA feed only the curve head (rating head untouched -> curve-side
gate). The metric scores the pooled pressed value at the label t; the proposal's claim is the
same shape as lambda 0.1 -> 0.2's +0.00048: making the counterfactual family coherent
regularises the shared (w, S, d) readout. Skeptical note: this is a regulariser story, not a
"fix an error" story; the accept bar is 0.0001 and lambda 0.3 already showed MORE weight on the
same-t constraint does not pay -- so the lever is only alive if the OTHER-t ordering is
substantially violated (screen).

**Cheap CPU screen (~30-60 min, no GPU).** Reuse `scratchpad/spacing_screen/monotonicity_probe.py`'s
instrument (deploy RNN, `predict_func(curve, t)` at fixed horizons). For every 20th review of 2
train-range users: snapshot the `RNNProcess` state (deepcopy), run the row 4 times with the grade
one-hot swapped and duration zeroed (exactly what `insert_probes` does), restore; read the 4
curves at {label t, 1 d, 7 d, 30 d, 180 d}; apply `rwkv/model/pava.py::pava_rectify` with the
checkpoint's `pava_theta`. Report: fraction of reviews whose button ORDER at t_h differs from the
order at the label t, and the pool fraction per horizon. **Kill:** crossings on < 3% of reviews at
every horizon (constraint already satisfied -> inert). Also report the median |R_Good - R_Hard| at
30 d: if the gap is < 0.01 the buttons barely separate and ordering is not the binding structure.

**Pre-registered band.** ahead **+0.0000 .. +0.0003**; imm inside the floor. Abort: ahead worse by
> 0.00015 (the hinge fights the same-t PAVA pooling -- lambda 0.02 retry once, then close).
P-diag: the crossing rate on the CANDIDATE must fall by >= 50% vs the control (engagement), and
`pava_pool_frac` at the label t should not rise.

**Not redundant with:** the killed spacing-effect lever (monotonicity in REVIEW COUNT; this is in
BUTTON at fixed t, PAVA's own axis), monotone-in-t (given by construction), lambda sweeps (same-t
weight). No relation to the calendar-aware curve (killed).

---

## 3. Probe density 0.08 -> 0.20, nothing else  -- rank 3

**Family:** probe / target sampling. **Provenance:** invented (the domain anchor is the deploy
contract: the served quantity is the pooled PRESSED curve with the current duration zeroed).

**Mechanism.** `RWKV_PROBE_DENSITY` (`prepare_batch.py:29`) selects 8% of eligible real rows to
receive 4 counterfactual rows. Only those rows train the deploy quantity (`_pava_probe_loss`,
lambda 0.2); the other 92% train the unrectified, duration-IN curve -- a quantity deploy never
serves (the mode-2 diagnostic prices the duration mismatch at +0.001451 ahead, ~70% of the deploy
penalty). Iter 33 moved the WHOLE objective onto probes at density 1.0 and lost, but bundled
three changes (probe-only, MAX 32768 -> 16384, and 23.5% of rows losing all supervision); the
single-variable coverage lever was never run. At density 0.20 the probe rows are 61% of real rows
instead of 24%: the PAVA gradient's variance falls 2.5x at the SAME lambda (mean-reduced loss, so
the weight does not move -- lambda 0.3 showed more WEIGHT does not pay; this is more COVERAGE),
and 2.5x more rows see the duration-zeroed input.

**Why AHEAD.** Probes touch only the curve head (curve-side gate). Cost: rows +30% over today
(1.61/1.24), so WS ~4.2 h instead of 3.2 h; eval unchanged; KD-off lineage, so no dump rebinding.
Zero code.

**Cheap CPU screen.** None can kill a coverage lever, and I will not invent one: the pre-registered
abort is the COST line. What CAN be checked in minutes: the PAVA loss trace's step-to-step
variance on the realcyc WS log (`pava_loss_avg` is in the trace since iter 23) -- if its
coefficient of variation is already < 0.1, variance reduction has nothing to buy and the lever
rests only on the input-robustness half.

**Pre-registered band.** ahead **+0.0000 .. +0.0002**; imm inside the floor. Abort: ahead worse
by > 0.0001 or a WS slowdown > 40%. **Partial overlap, stated:** probe rows ARE duration-zeroed
copies, so this is duration dropout restricted to the PAVA path -- PROPOSALS rank 4 (queued, not
run). If rank 4 runs first and wins, re-price this one on its result; if this runs first, it
bounds rank 4's PAVA-path share.

---

## 4. Probe the FIRST review of a card -- train and rectified eval  -- rank 4 (a deploy-correctness item; metric gain not expected)

**Family:** probe / target sampling + three-way parity. **Provenance:** invented; domain anchor is
FSRS's S0 block: initial stability by FIRST rating spans **0.11 d (Again) .. 11.8 d (Easy)**
(`fsrs_v7.py init_w[0..3]`), a 100x range -- the first press is the single most button-sensitive
point of a card's life.

**Mechanism.** `insert_probes` excludes a card's first in-chunk real row (`elig = real & has_lab &
~first_mask`, `prepare_batch.py:119-121`); the exclusion exists for the `probe_query` join, which
only `RWKV_PAVA_PWEIGHT` consumes (null at iter 24; the champion pools uniformly). Consequence at
eval (whole-history chunks of 2,097,152 rows, so first-in-chunk == genuine first review): the
ahead prediction for review 2 -- emitted at the first review -- is scored UNRECTIFIED with the
first review's duration IN, on ~16% of rows (share among SCORED rows to be measured; the earliest
fold dilutes it). At deploy that same interval comes from the rectifier with the duration zeroed.
So on ~1/6 of scored rows the rectified metric and the deploy contract disagree, and the
rectifier has never been trained where the buttons matter most. Change: drop `~first_mask` from
`elig`, emit `probe_query = -1` for unpairable targets and skip the `pairable` filter for them
(the filter's reason -- "a first review cannot be probed because the imm task needs a prior
review" -- is about the imm QUERY row, which uniform pooling never reads). Train and eval both
change; the Rust side already handles a first press (it rectifies from any state).

**Why AHEAD.** Curve-side only. Honest expectation: on a FAIR rect-to-rect pair the candidate may
gain a little from training PAVA on first reviews; the re-scored baseline number itself moves
WORSE by ~(0.16 x duration share) ~ +0.0002, which is the deploy truth being counted, not a
regression. Requires re-scoring the reference checkpoint under the new eval (eval only, ~4 h GPU,
no training) before the gate; both numbers must be reported.

**Cheap CPU screen (~20 min).** (a) parquet: share of EQUALIZED rows whose emitting row is a first
review (needs `label_filter_db_id_e2s` keys; 3 users); (b) deploy RNN on 2 users: at each first
review run the 4 counterfactual first presses from the fresh card state and count button-order
violations at the label t -- if PAVA pools on < 2% of first reviews AND the share in (a) is < 5%,
the metric effect is negligible and this is a contract note, not a run.

**Pre-registered band.** vs the re-scored baseline: ahead **-0.0001 .. +0.0002**; imm exactly
unchanged (probes have `has_label=0`, query rows untouched). This changes what the gate SCORES on
16% of rows -> **Andrew's call** before it runs, per the deploy-contract rule.

**Not redundant with:** iter 33 (withheld duration everywhere at density 1.0), duration dropout
(rank 4), the probe pairing fix of 2026-09-01 (which only DROPS unpairable targets).

---

## 5. Chunk-continuous training -- carry entity states across a user's chunks  -- rank 5 (screen first; probably small)

**Family:** training pipeline / state initialisation / data ordering. **Provenance:** invented;
the plan and the verified stateful WKV kernel already exist (`optimization/STATEFUL_BPTT_PLAN.md`
steps 2-3, shelved as a SPEED lever, never measured as an ACCURACY one).

**Mechanism.** Train chunks are `MAX_BATCH_SIZE = 16384` rows with zero state at every chunk start
(`data_processing_train_5k_h1_id5.toml`); eval chunks are 2,097,152 rows, and deploy is the whole
history. So training teaches the card/note/deck/user streams from windows that start cold
mid-history, while eval and deploy never do -- a section-9-shaped divergence. `get_groups`
shuffles chunks freely (`train_rwkv.py:290-311`); the carry needs a user's chunks in time order
with detached state hand-off (truncated BPTT), batching across users at the same chunk index.

**Why AHEAD.** Both heads read the trunk; the domain claim is that a mature card's memory state
is better read from its own 20-review history than from `elapsed_cumulative`-style summaries the
cold window forces the model to lean on. **The data screen already lowers the prior:** the
previous review is visible 88% of the time and 77% of a card's prior reviews sit inside the
window (per-user min 36%). The divergence is real but not large, and the plan's own note says a
32768-chunk cold re-baseline beat a 66000-chunk champion -- longer visible history did not pay
there. Rank 5 for that reason, and because it is the only multi-day engineering item here.

**Cheap CPU screen (~1 h, decisive).** Deploy RNN on 3 train-range users (the spacing-screen
instrument), resetting ALL five stream states every W rows for W in {4k, 8k, 16k, 32k, inf}; plot
ahead logloss vs W. **Kill:** inf vs 16k differs by < 0.0002 ahead -- then the model does not
exploit history beyond one training window, and the carry cannot be evaluated cheaply (it may
still teach the model to use it, but that claim needs the full build and the prior says no).
Second read: if the curve is still falling at 32k -> inf, the lever is alive.

**Pre-registered band.** ahead **+0.0000 .. +0.0003**, imm same sign (both-modes rule: trunk-wide).
Abort: any regression > 0.0001 (a detached carry changes the optimisation's effective sequence
length; a loss here means the cold windows were acting as a regulariser).

**Not redundant with:** the shelved stateful-BPTT SPEED question (measured null), MAX sweeps (the
batch dim, not the state), the deck tree (a new scope; this adds none).

---

## Ranking summary (expected ahead gain, one line each)

| # | lever | family | code | GPU cost | ahead band | kill screen |
|---|---|---|---|---|---|---|
| 1 | ordinal curve target from the k+1 rating | objective | ~40 lines, 4 params, train-only | 1 run | +0.0001..+0.0004 | data half PASSED (1.4x, 14/18 users); model half 45 min |
| 2 | multi-horizon button ordering on probes | curve regulariser | ~30 lines, train-only | 1 run | +0.0000..+0.0003 | crossing rate < 3% -> dead (30-60 min) |
| 3 | probe density 0.08 -> 0.20 | probe sampling | 0 lines | 1 run (+30% WS) | +0.0000..+0.0002 | none (cost line) |
| 4 | probe the first review (train + eval) | probe sampling / parity | ~15 lines | re-score + 1 run | -0.0001..+0.0002 vs re-scored base | share < 5% and pooling < 2% -> note only |
| 5 | chunk-continuous training | pipeline / state | multi-day | 1 run | +0.0000..+0.0003 | reset-cost curve flat past 16k -> dead |

Skeptic's notes. (i) 1 and 2 are curve-side and may be stacked if both pass alone; do not bundle
them in one run (the iter-33 lesson). (ii) 1 is the only one whose mechanism has a measured
precedent at this trunk (KD's variance channel, +0.0019 total over four accepts); everything else
is a regulariser or a coverage story against a 0.0001 bar. (iii) 4 is filed for correctness, not
for the number -- if Andrew declines the contract change, drop it without a run.
