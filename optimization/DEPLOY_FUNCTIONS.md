# The two functions CPU optimization is measured on

Andrew, 2026-07-26: the deploy-speed work is anchored on **named functions from the Anki fork** —
one that computes the five per-entity states, one that computes p(recall). This file records
exactly which they are.

**Only `review_features` is actually SCORED** — see [`DEPLOY_SPEED_LOG.md`](DEPLOY_SPEED_LOG.md),
which holds the single table. Andrew's call once it became clear the two overlap (§⚠ point 1
below): `retrievability_head` is a sub-computation of `review_features`, so a second table would
measure part of the same work twice. It stays documented here for reference.

Source: `vendor/jschoreels_anki/rust/mod.rs` — the READ-ONLY reference copy of
[github.com/JSchoreels/anki](https://github.com/JSchoreels/anki), an Anki fork that ships the old
2.76M RWKV as a live scheduler. Verified against the file, not from memory.

## 1. States — `review_features`

```rust
fn review_features(&self, features: &[f32], state: SrsStateRef<'_>) -> ReviewHeads
```
`vendor/jschoreels_anki/rust/mod.rs:2944`

One call consumes one review and produces **all five stream states**, chained in arch order —
`card → deck → note → preset → global`:

```rust
let x = self.feature_mlp(features);
let (x, card_state)   = self.modules[0].run(&x, state.card);
let (x, deck_state)   = self.modules[1].run(&x, state.deck);
let (x, note_state)   = self.modules[2].run(&x, state.note);
let (x, preset_state) = self.modules[3].run(&x, state.preset);
let (x, global_state) = self.modules[4].run(&x, state.global);
```

Note the order is card→**deck**→**note**→preset→global, matching our `architecture.py` module
order (`card_id, deck_id, note_id, preset_id, user_id`) — *not* the `RWKV_SUBMODULES` order. This
is the same trap AGENTS.md §"ERRATUM" records; the fork gets it right, and so must any comparison
against it.

**Unit: states/s.** One call = 5 stream states. Count *calls* per second and label it states/s
(the convention already used in `cpu_speed_log.md` §"What the throughput numbers MEAN").

**Our counterpart:** `Model::review` (candle, B=1) and `FastModel::review_batched` (the default
fast path) in `rust/rwkv-infer/src/{model,fast}.rs`. Both return the five new stream states plus
the heads, so they are the like-for-like target.

## 2. p(recall) — `retrievability_head`

```rust
fn retrievability_head(&self, prehead_x: &[f32]) -> f32 {
    1.0 - self.button_probabilities_head(prehead_x)[0]
}
```
`vendor/jschoreels_anki/rust/mod.rs:2940`

p(recall) = **1 − P(Again)**, off the rating head. `button_probabilities_head` (`mod.rs:2930`) is
`head_p_0 → relu → p_linear → softmax`, and index 0 is Again.

**Unit: probabilities/s.** One call = one p(recall).

**Our counterpart:** `Model::imm_prob` / `FastModel::imm_prob` — `1.0 - softmax(out_p_logits)[0]`,
the identical quantity. This is our benchmark's `imm` mode.

## ⚠ Three things to keep straight when measuring these

1. **They overlap — do not add their costs.** `review_features` computes retrievability *inline*
   (`let retrievability = 1.0 - button_probabilities[0];`), so `retrievability_head` is a
   sub-computation of it, not a disjoint stage. Timing them separately is still meaningful (they
   answer different product questions: building state vs re-scoring a due queue at fixed state),
   but their rates cannot be summed into a per-review total.
2. **The two rates differ by orders of magnitude, by design.** `review_features` runs the whole
   5-stream recurrence; `retrievability_head` is two small matmuls plus a softmax on an existing
   prehead vector. A shared y-axis or a single "speedup" figure across both would mislead.
3. **`prehead_x` is an input, not state.** `retrievability_head` can be benchmarked in isolation
   from a fixed prehead vector — which makes it the cleaner target of the two, since its timing is
   independent of state chaining, entity routing and the stored curve.

## 7. THREE REPLAY RULES FROM THE CLANKI REVIEW OF THE SHIPPED FORK (2026-09-24)

A read-only review of the Anki fork's RWKV code (Clanki session, `clanki-logs/reviews/algo-2026-09-24/`) found
three places where deploy departs from training. They are about the OLD shipped model, but each is a rule the new
builder must follow too.

1. **Replay the reviews of DELETED cards.** Training keeps them: the dataset builder never checks the cards table,
   and `data_processing.py:313` left-joins cards, so a deleted card's reviews stay with note/deck/preset missing and
   the `*_id_is_nan` flags set (~11% of rows over 7 sampled users; 23.8% in Andrew's collection). They update the
   user stream and the card stream. A replay that skips them gives every later prediction a state training never
   produced.
   ⚠ **WHICH placeholder entity a missing id becomes DEPENDS ON THE MODEL VERSION** (Clanki PR 292 builds the
   old-model rule; 2026-09-24). The OLD model was trained by srs-benchmark, which cast ids to int32, so
   `ID_PLACEHOLDER + card_id` and `ID_PLACEHOLDER` both SATURATED to INT32_MIN: ONE shared note, deck and preset
   for every missing id. The NEW model is trained on our int64 pipeline (since 2026-08-21): a missing NOTE is
   `ID_PLACEHOLDER + card_id`, i.e. each deleted card is its OWN note stream, while a missing deck and a missing
   preset still share one placeholder each. The fork's encoder must key this rule on the model version.
2. **An interval comes from the curve of the card's last REAL row, never from the query row.** The query row
   exists for the immediate prediction (rating head, before the answer). The ahead loss trains the curve of the
   REAL row after the answer, which predicts the next review. The query-row curve gets no loss at all. Measured in
   the fork: S90 off by a median factor of 2.2 and R 0.939 against a 0.913 pass rate.
3. **An entity's id code is a deterministic function of the entity id, and a read-only query never mutates state.**
   A code drawn from an RNG in query order makes the same card/note/deck get different codes live and after a
   rebuild.

## ⚠ Licensing

`vendor/jschoreels_anki/` is **AGPL-3.0-or-later** (see its `NOTICE.md`). *Naming* these functions
and benchmarking against their structure carries no licensing consequence. **Copying any of their
implementation into `rust/rwkv-infer` would make our engine AGPL-derived** — fine for shipping
inside Anki, but it must be flagged to Andrew first and carry provenance comments. That applies in
particular to the AVX2/FMA `x86_simd.patch`, which is the most tempting thing in there.

---

## ⚠⚠ FILTERED (CRAM) DECKS: A DEPLOY REQUIREMENT NO GATE CAN CATCH (Andrew, 2026-08-26)

Andrew's question after we measured that filtered decks are absent from the training data:
*"since filtered decks aren't in the training data how do we make sure they are handled
correctly during deployment?"* They are absent from the data only because Anki's export
resolves them away; they are very much present at deploy time.

**WHAT TRAINING ACTUALLY SAW** -- three facts, checked in code, that together make the
requirement precise:

1. **Filtered REVIEWS are in the training data.** `filter_revlog` drops only
   `review_kind == 3 AND ease_factor == 0` (filtered-deck *rescheduling* artifacts). A real
   review answered inside a filtered deck is kept
   (`scratchpad/dataset_id/build_parquet_id.py:58-59`).
2. **They are attributed to the HOME deck.** Anki's export resolves `odid`, verified by
   measurement rather than by reading Anki: cards referencing a deck absent from the decks
   table are 7 of 1,608,074 (0.0004%), and no deck carries the `preset_id == 0` filtered marker
   (`scratchpad/hybrid100k/filtered_deck_probe.py`).
3. **Their filtered-ness is INVISIBLE to the model.** `review_kind` becomes `scaled_state`,
   feature column 22, and the champion sets `RWKV_ZERO_FEATURES=22`. The rebuild drops the
   column outright.

=> **Training's view is "a cram review is an ordinary review, in the home deck."** The deploy
requirement is therefore to reproduce exactly that, not to add special handling.

**THE REQUIREMENT, for whoever writes the Anki integration:**

* Resolve the home deck before filling `ReviewInput.deck_id`:
  `if odid != 0 { odid } else { did }`.
* Resolve the **preset the same way** -- the home deck's preset, not the filtered deck's config.
  A filtered deck schedules from its own config, which is exactly the value NOT to pass.
* Do **not** create a deck entity for the filtered deck, and do **not** flag cram reviews.
  Training cannot see the flag, so supplying one is a divergence, not an improvement.

**WHY NOTHING WILL CATCH THIS.** `review_features` takes `deck_id` as an INPUT
(`ReviewInput.deck_id`, mod.rs:357); the engine never looks it up. Pass the filtered deck's id
and every call still succeeds. The failure is silent and expensive: each cram session invents a
brand-new deck entity whose stream state starts at zero, discarding the accumulated home-deck
state, and Anki then deletes the filtered deck so that state is thrown away. Predictions are
simply worse during and after cramming, with no error anywhere.

**The reference fork demonstrates the correct resolution but only in its SCAN path** --
`scan_bench_current_deck_id_sql` (mod.rs:6715-6721) emits
`case when c.odid != 0 then c.odid else c.did end`, with a fallback to bare `c.did` on schemas
lacking the column. Its LIVE path delegates to whoever constructs `ReviewInput`; every in-file
construction site is a test. So this is a pattern to copy, not a guarantee we inherit.
⚠ Note also mod.rs:6696 passes `preset_id: Some(row.deck_id)` in that bench path -- a deck id
used as a preset id. Their code, their bench, but do not copy that line.

**HONEST RESIDUAL, not a defect:** the model cannot distinguish deliberate cramming from
studying ahead, because the only column that marks it is zeroed. It infers what it can from the
short elapsed time, which IS an input. That limitation is shared by training and deploy, which
is what makes deployment *consistent*; whether cram deserves distinct treatment is a research
question and would need the state column back.


---

## 4. ⚠ THE GEN-5 FEATURE VECTOR HAS NO DEPLOY-SIDE IMPLEMENTATION (2026-09-09)

`review_features` takes the input vector as a SLICE, so the caller builds it -- and nothing in this
repo builds it for deploy. `rust/rwkv-infer` computes no feature (grepped: zero hits for `tod_sin`,
`sibling`, `creation_batch`, `tenure`, `deck_depth`, `cyc3_`, `dow_sin`), and
`run_as_rnn.RNNProcess.get_tensor` reads `CARD_FEATURE_COLUMNS` off a row `get_rwkv_data` already
produced. Both consume; neither derives.

**The §9 half of that is GOOD NEWS: there is no second implementation, so nothing can diverge.**
Under `RWKV_ID_FEATURES=1` the trace export takes the shared `get_rwkv_data` path, which is why the
gen-5 parity trace came back self-contained at 0.000e+00.

**⚠⚠ CORRECTION 2026-09-12: TRUE FOR THE 47 NEW COLUMNS, FALSE FOR THE OLD ONES, AND TWO HAD
DIVERGED.** `run_as_rnn.RNNProcess.add_same` DOES derive the published-era history columns a second
time, online, exactly as a live scheduler must (the cumulative elapsed pair and its phases, the three
missing-id flags, day_offset_diff, diff_new_cards, diff_reviews, cum_*_today, the scaled intervals).
`scratchpad/final_layout/deploy_vs_lmdb.py` compares that vector with the LMDB row by row: two had
drifted from training, and "self-contained at 0.000e+00" could not see it, because a trace is checked
against the Python path that BUILT it, never against the data the model trained on.
1. **The cumulative elapsed columns summed the -1 "no previous review" sentinel.** Training fixed this
   on 2026-08-19 (`data_processing.py`, "THE SENTINEL MUST NOT BE SUMMED"); the deploy half never got
   the fix. `scaled_elapsed_days_cumulative` was off on 39-73% of VAL rows, by up to 0.31 sigma.
2. **The missing-id flags came out 0 on every missing-card row** (~26% of all `-id` reviews), because
   `add_id` tested `np.isnan(id)` on ids `get_rwkv_data` had already filled. Training has 1 there.
Fixed in `run_as_rnn.py` (2026-09-12): every column now matches the LMDB within bf16 on five VAL users
(~130k rows); on user 5022 the deploy imm LogLoss moves from +0.000167 to +0.000007 of the GPU eval's.
`RWKV_DEPLOY_CUMSUM_LEGACY=1` restores the old sum, for models trained on an LMDB built before
2026-08-19. **THE ANKI BUILDER MUST DO THE SAME:** (1) `elapsed_*_cumulative` = -1 on a row whose own
`elapsed_*` is -1, else the sum of the card's REAL gaps up to and including this row, each column
masked on its own sentinel; (2) a missing-id flag = 1 iff the review's card row is absent (a deleted
card), and the note id then gets the per-card placeholder. `reference_realcyc`'s traces predate the fix
and carry both errors; the w10lf line uses `reference_final_inputs/` (checked against the LMDB by
`check_input_traces.py`).
**THE REUSABLE CHECK:** a deploy feature builder is right iff its vector equals the LMDB's for the same
`review_th`. Run `deploy_vs_lmdb.py` after any change to `add_same`, `get_rwkv_data` or the builder.

**The other half is unwritten phase-7 work.** Gen 5 adds **47 card-feature columns** a live Anki
scheduler must produce from its own database, and none exists outside `rwkv/id_features.py`.
**It does NOT touch the frozen 9 B/card / 27 B/note budget** -- the extra quantities are collection
facts Anki already stores (card ids are creation timestamps; the revlog holds every review), so
they are QUERIES, not serialised state. Only `tod_dev` wants running state, it is per-USER, and the
global scope is the one this project always allowed to grow. Per-column breakdown, the two
non-O(1) items, and the sentinel that a deploy implementation must reproduce exactly:
`scratchpad/feataudit/DEPLOY_DEBT.md`.

### 4b. `scaled_cum_new_cards` -- ONE COUNTER, AND IT IS THE FIRST COLUMN DEPLOY MUST *MAINTAIN* (2026-09-20)

`RWKV_EXTRA_FEATURES=cum_new_cards` appends input column 104: the number of distinct cards the user has
reviewed at least once, INCLUDING the current card when this review is its first. Andrew asked for it as a
new input; it needs no LMDB rebuild, because the value is a pure function of `(user_id, review_th)` and the
row already carries `review_th` (`rwkv/extra_features.py`).

**The Anki-fork rule, and it is one line of state:**

```
per collection:  cum_new_cards : u64          # distinct cards ever reviewed
at each review:  if this card has no prior review { cum_new_cards += 1 }     # BEFORE building the vector
                 feature = (ln(1 + cum_new_cards) - 8.882881) / 1.295293
```

* **Legal at show time.** A scheduler knows it is showing a new card, so the increment happens before the
  prediction, not after it. That is also what training does -- `data_processing.py:364` is
  `is_first_review.cumsum()`, inclusive of the current row.
* **The imm query row and the PAVA probes take the SAME value as their real row**, with no special case:
  in training they share its `review_th`, and in deploy `run_as_rnn` updates `card_set` only in
  `process_row`, so `imm_predict` reads the pre-update count. Checked, not assumed.
* **Constants** are the train half only (users 1-5000, 357,460,180 rows): `ln1p` mean 8.882881,
  std 1.295293. A collection with no reviews yet reads -6.86, which is the intended cold value.
* **It is NOT per-card state** -- one integer per collection, so the frozen 9 B/card / 27 B/note budget is
  untouched, exactly like the running statistics section 5 and section 6 already require.
* **Cards deleted, or reviews deleted, must not decrement it.** The quantity is "ever reviewed", and a
  counter that goes down is a quantity training never saw. On a collection whose revlog is rebuilt or
  imported, recompute it as the number of distinct card ids in the revlog rather than replaying.

Checks that must pass after any change here: `scratchpad/newcards/smoke_extra_features.py` (inert when off,
values equal to a parquet recomputation at bf16, query rows inherit) and
`RWKV_EXTRA_FEATURES=cum_new_cards python scratchpad/final_layout/deploy_vs_lmdb.py` (the online counter
against the side table, per column, real and query rows).

### 4c. THE EIGHT ROW COLUMNS AND THE SIBLING PAIR -- WHAT THE ANKI BUILDER MUST MAINTAIN (2026-09-23)

`RWKV_EXTRA_FEATURES` also carries eight per-ROW columns (accepted with 102 `rowfeat`, directed) and the two
sibling columns (under test in `s64hsib`). Training reads them from precomputed tables
(`scratchpad/rowtables/`); the Python deploy path (`run_as_rnn`) computes them online and
`deploy_vs_lmdb.py` shows 0 disagreeing columns against the tables on real AND query rows. The Anki builder
must compute exactly the rules below. Source of truth: `rwkv/extra_features.py` (`_build_rowfeat`,
`ROWFEAT_STATS`) and `scratchpad/rowtables/build_sibtable.py` (`sibling_grades_loop`).

**Time base.** `t` = the review's SHOW time in whole epoch seconds (`review_time // 1000`, i.e. answer time
minus `taken_millis`). A live scheduler has this exactly: it is `now()` when the card is shown. Every value
uses reviews STRICTLY BEFORE the current one, so all ten are legal at show time.

**Per-collection state (NOT per card; the 9 B/card / 27 B/note budget is untouched):**

```
prev_t, sess_open_idx, sess_open_t, n          # n = reviews so far
ring of recent t (enough for 3600 s)           # for roll10 / roll60
sum cos(th), sin(th), cos(2th), sin(2th)       # th = 2*pi*(t mod 86400)/86400, UTC
last_day, run_len, streak_before_today         # streak
sum_rating                                     # grade_mean
sess_id; per note: (sess_id, last_card, last_rating, other_rating)   # sibling; only this sitting's notes
```

| column | raw value for review i (reviews 0..i-1 only) | then |
|---|---|---|
| `sess_pos` | reviews already done in this sitting; a sitting opens when `t - prev_t > 1800` (the first review opens one) | ln(1+x), z-score |
| `sess_secs` | `t - t(sitting open)` | ln(1+x), z-score |
| `roll10` | count of earlier reviews with `t_j >= t - 600` | ln(1+x), z-score |
| `roll60` | same with 3600 | ln(1+x), z-score |
| `phase_r` | `m = max(i,1)`, `R = |sum e^{i th}| / m`, value = `sqrt(max((m R^2 - 1)/max(m-1,1), 0))` (Rayleigh bias correction) | z-score |
| `phase_r2` | `|sum e^{2i th}| / m` | z-score |
| `streak` | consecutive UTC days (`t // 86400`) with a review, counted through the day of the PREVIOUS review; a gap of 2+ days resets to 1; 0 for review 0 | ln(1+x), z-score |
| `grade_mean` | `sum_rating / max(i,1)` over earlier reviews (ratings 1..4) | z-score |
| `sib_grade` | see below; no sibling -> 0.0 after scaling | z-score |
| `sib_none` | 1.0 if no sibling, else 0.0 | as is |
| `sibgap_none` (arm `s64hsg10`, 2026-09-25; **NOT ADOPTED 2026-09-26, iteration 113: a clean null -- do not port**) | 1.0 if this row's scaled sibling gap is the 0.0 "no earlier review of another card of this note" sentinel, else 0.0 -- i.e. 1.0 exactly when the sibling gap is undefined | as is |

`phase_r`, `phase_r2` and `grade_mean` on the collection's FIRST review have no history: feed **0.0**
(the scaled training mean). Constants (train half, users 1-5000, row-weighted; mean / std):
`sess_pos` 4.163197 / 1.424215, `sess_secs` 6.894923 / 1.687621, `roll10` 3.283222 / 1.111573,
`roll60` 4.097617 / 1.283437, `phase_r` 0.440765 / 0.164353, `phase_r2` 0.268032 / 0.152146,
`streak` 2.607561 / 1.485916, `grade_mean` 2.608538 / 0.307149, `sib_grade` 2.718832 / 0.787373.

**The sibling rule, exactly (`sibling_grades_loop`).** Sittings use the same 1800 s cut. For a card WITH a
note (a card whose note id is missing in the export counts as having NO note -> `sib_none` = 1): look up
the note's entry for the CURRENT sitting. No entry -> no sibling. If the entry's last review was a
DIFFERENT card, the sibling grade is that review's rating; if it was THIS card, it is the entry's
`other_rating` (the latest different-card grade before it, 0 = none). After the review, write
`(sess_id, this card, this rating, sibling grade just computed)`. Entries from earlier sittings are dead
and can be dropped.

**Query row and PAVA probes** take the same values as their real row (they share its `review_th`). **Update
order:** build the vector from the state, predict, and only after the ANSWER advance the state (the
answer's rating feeds `grade_mean` and the sibling entry).

Checks after any change here: `scratchpad/rowtables/smoke_rowfeat.py`, `scratchpad/rowtables/leak_check.py`,
and `RWKV_EXTRA_FEATURES=<the columns> python scratchpad/final_layout/deploy_vs_lmdb.py`.

## 5. ⚠ THE CLOCK RULE FOR ROWS THAT PREDICT AT SHOW TIME (2026-09-10, `RWKV_CLOCK_AT_PREV_ANSWER`)

The imm query row and the four PAVA button probes predict BEFORE the answer. In the -id data their
clock columns were measured to the reconstructed show time `id - taken_millis`, which Anki's 60 s
duration cap and timer resets make late by the current review's excess -- a leak of the outcome
(`scratchpad/leak/PREREG.md`). A model trained under `RWKV_CLOCK_AT_PREV_ANSWER=T` expects, on those
rows only:

* `delta = now - (last answer on ANY card)` if that is below `T`, else `0`;
* `t_since_any_review` = 0 when `delta > 0`; the same-card interval, its cumulative, user tenure,
  deck age and sibling gap each minus `delta` (floored at 0, undefined sentinels untouched); the
  daily sin/cos pairs (interval, cumulative, time of day, time-of-day deviation) rotated back by
  `delta`;
* **if `delta` is larger than the seconds since UTC midnight, the moved time is on the PREVIOUS UTC
  day** (added 2026-09-10): dow, doy and the review-time real cycles rotate back by one day
  (`2*pi/period`) and `is_weekend` is re-derived from the rotated weekday. The simplest correct
  implementation computes EVERY clock column of these rows from `now - delta` instead of `now`;
* the curve evaluated at `t - delta` wherever `t` is the time since the card's previous answer.

Two groups stay unshifted, because a stored training row cannot recompute them: the Anki-day
columns (elapsed_days and its cumulative, day_offset_diff, cum_*_today) and the creation-batch
counts. So the deploy builder must compute them at `now`, as the database did. Measured on users
5001-5040 (`scratchpad/leak/residual_channels.py`), they would change on 0.0036% and 0.021% of
query rows, with no measurable outcome signal.

REAL rows (a finished review) are NOT shifted: they are rebuilt from the revlog, excess included,
exactly as in training. The rule is one function, `rwkv/clock_fix.py` (`shift_features`,
`ahead_t`), called by training and eval (`prepare_batch.prepare`) and by the Python deploy path
(`run_as_rnn.imm_predict`, the trace exporter). The Rust engine consumes rows its caller built, so
**the Anki fork's feature builder must apply the same rule** to the query row and to the row it hands
`button_intervals`. At deploy the in-session gap is ~0 anyway, so the rule mostly confirms what the
scheduler sees; it matters for a short real break, which it must zero exactly as training did.
Default off: a model trained without the flag expects the raw values.

## 6. ⚠ THE DURATION RULE FOR ROWS THAT PREDICT AT SHOW TIME (2026-09-15, `RWKV_PROBE_IMPUTE_K`, R1)

The imm query row and the four PAVA button probes predict BEFORE the answer, so the current review's
`duration` does not exist yet. The deploy contract (section 3 of AGENTS.md's DEPLOY CONTRACT) writes
`scaled_duration = 0.0` on those rows -- the same constant for every card of every user. A model
trained under `RWKV_PROBE_IMPUTE_K=k` expects, on those rows only:

    scaled_duration = w * median(this card's PAST scaled durations)
                    + (1 - w) * median(this user's PAST scaled durations),   w = n_card / (n_card + k)

* "past" = reviews the user has ANSWERED before this one is shown -- never the current review, never
  a later one, never a query/probe row. Both medians are running medians in review order;
* a card with no earlier review: `w = 0`, the user median; a user with no earlier review: `0.0`;
* the SAME value on the query row and on all four probe rows of a review (duration is spent before
  the press, so it cannot depend on the button);
* `scaled_duration = (log(10 + duration_ms) - duration_mean) / duration_std`; the median is taken in
  scaled space (equivalently, the scaled median of the raw durations -- the scaling is monotone), with
  the mean of the two middle values on an even count;
* REAL rows (a finished review) keep their real duration, as always.

One implementation, `rwkv/probe_impute.py` (`OnlineImputer`: `value(card)` before the review,
`push(card, scaled_duration)` after it), is used by training and eval (`prepare_batch.prepare`) and by
the Python deploy path (`run_as_rnn.imm_predict` / `process_row`). The Rust engine consumes rows its
caller built, so **the Anki fork's feature builder must keep the two running medians per collection**
(the per-card one is small; the per-user one is a running median over the whole revlog -- two heaps
or a histogram over the bf16 codes) and apply the rule to the query row and to the row it hands
`button_intervals`. Default off: a model trained without the flag expects 0.0.
