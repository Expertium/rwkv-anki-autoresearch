# Interval definition: end-to-END vs end-to-START — handoff

Andrew, 2026-08-30: *"find the piece of preprocessing code where interval length is calculated as
end-to-end or end-to-start ... so that I can give it to another Claude instance and let it run
FSRS-7 end-to-end vs end-to-start."*

Self-contained. Everything needed to run that comparison is below; nothing else in this repo is
required.

---

## 1. The definitions

A revlog row is written when the user **answers**, so `revlog.id` is the ANSWER time and
`duration` (`taken_millis`) is how long that review took. Write `show(k) = answer(k) - duration(k)`.

| name | formula | what it measures |
|---|---|---|
| **end-to-END** | `answer(k) - answer(k-1)` | the gap plus the current review's own duration |
| **end-to-START** | `show(k) - answer(k-1)` | the gap during which the memory actually decays |

    end_to_start = end_to_end - duration(k)

**Why end-to-start is the right quantity.** Decay starts when the user finished being shown the
answer last time — `answer(k-1)` — and the test happens when the card is next SHOWN, `show(k)`.
Anything after that is the user thinking, which is not elapsed forgetting time.

**⚠ There is a second reason, and it is stronger than precision.** `duration(k)` does not exist at
prediction time (the user has been shown the card and has not answered yet), and it **correlates
with the outcome**, because a review the user struggles with takes longer. So end-to-END feeds
every algorithm a prediction-time-unavailable, outcome-correlated quantity, hidden inside the
interval. That is the same issue our own deploy contract already handles by zeroing the most
recent review's duration.

---

## 2. ⚠ THE FORMULA DIFFERS BY DATASET — this is the trap

The two datasets store a **different timestamp under the same column name**, so the same physical
quantity needs a different subtraction. Applying one dataset's formula to the other is silently
wrong, not obviously wrong.

| dataset | timestamp column | it holds | end-to-start |
|---|---|---|---|
| **published `anki-revlogs-10k`** (what srs-benchmark uses) | `elapsed_seconds` (no timestamps) | ANSWER-to-ANSWER gap | `elapsed_seconds - duration(k)/1000` — subtract **THIS** review's duration |
| **`anki-revlogs-10k-id`** (ours, local) | `review_time` | the **SHOW** time (`id - taken_millis`) | `review_time(k) - (review_time(k-1) + duration(k-1))` — subtract the **PREVIOUS** review's duration |

Both give `show(k) - answer(k-1)`. They differ only because `-id` already moved the timestamp back
by one duration.

**For an srs-benchmark experiment, use the FIRST row.** `duration` and `elapsed_seconds` are both
columns of the public dataset, so this needs **no new data** — it is a one-line transform of what
everyone already has.

---

## 3. Where it plugs into srs-benchmark

`features/base.py::_process_time_intervals` is the ONLY place `delta_t_secs` is derived
(everything else matching `elapsed_seconds` in that tree is inside `.venv`):

```python
if (
    "delta_t" not in df.columns
    and "elapsed_days" in df.columns
    and "elapsed_seconds" in df.columns
):
    df["delta_t"] = df["elapsed_days"]
    if self.config.use_secs_intervals:
        df["delta_t_secs"] = df["elapsed_seconds"] / 86400
        df["delta_t_secs"] = df["delta_t_secs"].map(lambda x: max(0, x))
```

The end-to-start arm is one changed line:

```python
        # end-to-START: remove THIS review's own duration, which is not knowable at prediction
        # time and correlates with the outcome. `duration` is in MILLISECONDS.
        df["delta_t_secs"] = (df["elapsed_seconds"] - df["duration"] / 1000.0) / 86400
        df["delta_t_secs"] = df["delta_t_secs"].map(lambda x: max(0, x))
```

Three things that make this cheaper than it looks:

* It already sits behind `use_secs_intervals` (the `--secs` flag), so the new option goes beside an
  existing one rather than adding an axis.
* **The `max(0, x)` clamp is already there.** 0.559% of same-day rows have a duration exceeding the
  whole recorded gap, and the existing clamp handles them — no new rule to decide.
* **`delta_t` (days) must NOT be touched.** It is a calendar-day index difference matching Anki's
  scheduling semantics; "subtract a duration" is not well defined on a day index, and the effect at
  day resolution is ~0.001% anyway.

---

## 4. Measured effect size (40 stride-sampled users, 2.18 M reviews)

`scratchpad/hybrid100k/interval_def_effect.py`.

| | same-day (30.8% of rows) | longer interval (69.2%) |
|---|---|---|
| median gap | 485 s | 531,311 s |
| duration as a % of the gap, median | **1.70%** | 0.0012% |
| ...p90 | **13.83%** | 0.01% |
| ...p99 | **65.22%** | 0.07% |
| rows moving >= 10% | **13.9%** | 0.0% |
| corrected gap goes negative | **0.559%** | 0.000% |

**The effect is a TAIL, not a median.** 1.70% reads as dismissible; 13.9% of same-day rows moving
by a tenth or more does not, and those are the short learning-step reviews where short-term
modelling is actually differentiated.

**On long intervals it is numerically invisible**, so this cannot change the "Without same-day
reviews" table at all. Only the with-same-day table can move.

**Pre-registered prediction (recorded before running it):** aggregate LogLoss moves less than the
gap between adjacent rows of the with-same-day table, and no ranking changes. If that holds it is a
`--interval-def` flag and a footnote; if a ranking DOES move, it is a correction worth making
properly.

---

## 5. Our own live code, for reference

`rwkv/id_features.py::elapsed_end_to_start`. **Note it is gated on the DATASET, not on a flag** —
it runs only when `review_time` is present, which only the `-id` set has. So the current champion,
which trained on the published `train_db_5k_h1`, is on **end-to-END**; end-to-start applies to the
`-id` rebuilds.

```python
def elapsed_end_to_start(df):
    if "review_time" not in df.columns or "duration" not in df.columns or not len(df):
        return df
    answer_prev = (df["review_time"] + df["duration"]).groupby(df["card_id"]).shift()
    gap_s = ((df["review_time"] - answer_prev) / 1000.0).clip(lower=0.0)  # clamp BEFORE the cast
    out = np.floor(gap_s.to_numpy())
    no_prev = np.isnan(out) | (df["elapsed_seconds"] == -1).to_numpy()
    out[no_prev] = -1.0
    assert not np.isnan(out).any()
    df = df.copy()
    df["elapsed_seconds"] = out.astype("int64")
    return df
```

Four details that cost real debugging and are worth carrying:

1. **Clamp BEFORE the int cast.** Flooring -0.4 gives -1, which silently mints a fake "first
   review" — the sentinel is -1.
2. **The sentinel mask is "no known previous review FOR THIS CARD"**, which is NOT the same as
   `elapsed_seconds == -1` (that marks `state == 0`). A card whose first row in the frame is not a
   state-0 row gets NaN from the groupby shift. Measured 1 row in 3,817,339 over 40 users —
   vanishingly rare per row, certain across 5,000 users, and one NaN kills a whole user.
3. **Upstream's value for such a row is a CROSS-CARD diff** (it diffs the whole frame in protobuf
   order and only overwrites state-0 rows with -1), i.e. the gap to a *different* card's last
   review. The sentinel is the honest replacement, not a regression.
4. **`elapsed_days` is deliberately untouched**, for the reason in section 3.

Smoke: `scratchpad/features_rebuild/smoke_end_to_start.py`.

---

## 6. ⚠ CORRECTION (2026-08-30): the 0.559% clamp figure above is a FLOORING ARTIFACT

PR [open-spaced-repetition/anki-revlogs-dataset-builder#2] adds `--elapsed-end-to-start` to the
builder and reports end-to-start going negative on **2 rows in 2,306,229**, computed per card from
real start/end timestamps. Section 4 above says **0.559% of same-day rows**. Both cannot be right.

**There is a proof, not just a measurement.** For two consecutive reviews of the SAME card the card
cannot be shown before it was last answered, so `start(k) >= answer(k-1)`, i.e.
`elapsed_seconds >= duration(k)`. **End-to-start cannot be negative.** Every negative is therefore
an artifact, and the only question is which.

Split by magnitude (40 users, 2,179,548 real gaps, `negative_gap_origin.py` and the follow-up):

| negatives | count | share |
|---|---|---|
| in (-1, 0) -- **flooring** `elapsed_seconds` to whole seconds | 3,738 | **99.7%** |
| <= -1 s -- a cross-card protobuf-block-boundary diff | 12 | 0.3% |

`elapsed_seconds` is stored as whole SECONDS; subtracting a millisecond-precision `duration` pushes
the sub-second population just below zero. Measured 0.172% of rows, against PR #2's **0.222%** of
eligible rows landing in `[0, 1)` s -- the same population. The 12 rows flooring cannot explain
match the PR's 2-in-2.3M genuine overlaps in order of magnitude.

**=> The clamp is still required on the published set, but it is a QUANTIZATION artifact bounded by
1 second, not a physical overlap.** Section 4's 0.559% should be read as "0.17% of all real gaps,
shifted by under a second" and not quoted as a property of the interval definition.

⚠ **A method note, because my first attempt at this was invalid.** I tried to separate the causes by
testing whether the PREVIOUS ROW is the same card. It is not a valid test: upstream computes the
diff in PROTOBUF order (per-card blocks) and sorts by `review_time` AFTERWARDS, so row adjacency in
the stored frame says nothing about how `elapsed_seconds` was computed -- only 0.3% of rows are
adjacent-same-card. The magnitude split above works because it rests on the inequality, which needs
no assumption about ordering.

### Two implementation loci, and they are not competing

* **PR #2 -- in the BUILDER.** Correct at the source, full millisecond precision, no flooring
  artifact at all. Costs a dataset rebuild and re-upload before anyone can consume it.
* **This document, section 3 -- in srs-benchmark's feature derivation.** One line, works on the data
  everyone already has, and inherits the existing `max(0, x)` clamp. Carries the sub-second
  flooring artifact above, which is immaterial to a forgetting curve.

Use whichever fits the question. For a quick FSRS-7 A/B the second is enough; for a published
third table the first is the honest source.

---

## 7. Verified identical to PR #2 (Andrew's ask, 2026-08-30)

| our function | matches the PR? | note |
|---|---|---|
| `elapsed_end_to_start_published` | **yes**, semantically | operates on the published parquet's stored `elapsed_seconds`, which upstream computed with the same ungrouped diff and the same `state == 0` sentinel set. Residual difference is PRECISION only: that column is whole seconds and cannot carry the PR's millisecond endpoints. Bounded by 1 s; see section 6. |
| `elapsed_end_to_start` (`-id`) | **yes**, semantically | same formula `start(k) - end(k-1)`, same clamp, same sentinel. |

**★ THE ONE APPARENT DIFFERENCE IS REQUIRED, AND COPYING THE PR LITERALLY WAS MEASURED TO BE
CATASTROPHIC.** The PR uses a PLAIN frame `.shift()`; ours groups by `card_id`. That is not a
deviation:

* the PR runs inside the BUILDER, on the frame in **protobuf order** (per-card blocks), BEFORE
  `sort_values("review_time")` -- so its plain shift is per-card by virtue of the ordering;
* ours runs on the already-SORTED parquet, where the previous row is almost always a DIFFERENT
  card reviewed seconds earlier.

Changed to a plain shift to "match", the smoke reported **every gap bucket shortened by a median
100%, including gaps over a day** -- it would have silently destroyed every interval in the `-id`
dbs. Reverted. `scratchpad/features_rebuild/smoke_end_to_start.py` is what caught it, and the
post-revert numbers are the sane ones: 1-10 min gaps shorten 10.9%, 10 min-1 day 1.1%, over a day
0.0%.

**The general lesson: "identical preprocessing" means identical SEMANTICS, and two code paths that
run at different points in the pipeline need different code to compute the same thing.** A literal
copy across that boundary is a silent correctness bug, not a safe default.

---

## 8. ★★ THE FSRS-7 RESULT, AND WHY IT DOES NOT SETTLE THE DEPLOY QUESTION (2026-08-30)

Andrew ran it on all 10,000 users:

| FSRS-7 (Rust) | end-to-end | end-to-start | diff |
|---|---|---|---|
| LogLoss | **0.3179** | 0.3182 | +0.0003 |
| RMSE(bins) | 6.36% | 6.36% | 0.0000 |
| AUC | **0.7520** | 0.7515 | -0.0005 |

**End-to-start is slightly WORSE, and the pre-registered prediction in section 4 held** — the move
is far smaller than the gap between adjacent rows of the with-same-day table, and no ranking can
change. As a *benchmark* result this is a footnote.

### It is not a footnote for deployment, and the reason is measurable

Two readings fit those numbers and they imply opposite decisions:

* **A — end-to-END is genuinely more informative.** Keep it.
* **B — end-to-END LEAKS.** It is `answer(k) - answer(k-1)`, so it silently contains
  `duration(k)`: the length of the very review being predicted. That quantity does not exist at
  prediction time, and it correlates with the outcome. A larger interval lowers predicted
  retrievability — exactly the right direction on a review that was about to be failed. A
  benchmark that trains *and* scores both arms self-consistently **rewards** the leak.

**The test that separates them, and it needs no model.** Hold the end-to-start gap fixed, then ask
whether `duration(k)` still separates outcomes. Inside a narrow band of end-to-start gap the
elapsed forgetting time is already fully described, so duration carries **no legitimate information
about decay** there. Any residual predictive power is leak by construction.

`scratchpad/hybrid100k/duration_leak_probe.py`, 40 stride-sampled users, 2,179,548 real gaps,
40 quantile bins, single-threaded CPU:

| stratum | rows | AUC(dur) | **AUC(dur \| gap)** | shuffled floor | dur/interval, median |
|---|---|---|---|---|---|
| all | 2,179,548 | 0.5942 | 0.5989 | 0.4991 | 0.004% |
| **same-day** | 845,029 | 0.6092 | **0.6181** | 0.4996 | 0.959% |
| longer than a day | 1,334,519 | 0.5847 | 0.5866 | 0.5002 | 0.001% |

**READING B IS SUPPORTED.** At a fixed end-to-start gap, `duration(k)` predicts failure at
**AUC 0.618** against a shuffled-within-bin floor of **0.4996**. For scale, FSRS-7's entire model
scores AUC 0.752 — so the leaked quantity alone carries a large fraction of the discrimination the
model is being graded on. And it is injected exactly where it is material: it moves the interval by
≥10% on **11.1% of same-day rows** and **0.00%** of longer ones.

Both controls were necessary and both did their job. The shuffled arm shows the stratified
statistic has no finite-sample bias here (0.4991–0.5002). The long-interval stratum shows duration
predicts the outcome *everywhere* — which is why "duration correlates with failure" alone proves
nothing, and why the interval-share column is what bounds the leak's scope.

⚠ This does **not** prove the 0.0003 *is* the leak. It proves a leak channel of ample size exists
in exactly the right place and direction, so the sign of the benchmark result carries no evidence
about which definition is better *at deploy*.

**One more thing makes the FSRS case clean: FSRS-7 has no duration input.** In srs-benchmark only
`lstm_engineer.py` mentions an optional duration feature; FSRS consumes `delta_t` and ratings. So
end-to-end is the model's **only** duration channel, and closing it can only lose the leak.

### ★★★ AND FOR OUR MODEL THIS IS A TRAIN/DEPLOY DIVERGENCE — a §9 three-way-parity miss

Verified in the shipping Anki fork, `vendor/jschoreels_anki/rust/rwkv.rs:322`:

```rust
let elapsed_seconds = last_review_time.map(|last_review_time| {
    TimestampSecs::now().elapsed_secs_since(last_review_time).max(0) as u32
});
```

`last_review_time` is the previous revlog's timestamp, i.e. **answer(k-1)**; `now()` is evaluated
while the scheduler is choosing what to show, strictly **before** the user answers. So a live Anki
scheduler computes `show(k) - answer(k-1)`. **Deploy is end-to-START and structurally cannot be
anything else** — `duration(k)` has not happened yet.

Our champion trained on the published `train_db_5k_h1`, whose `elapsed_seconds` is
`answer(k) - answer(k-1)` = **end-to-END**.

| path | interval it uses |
|---|---|
| TRAINING | end-to-END (the dataset column) |
| EVAL | end-to-END (same column) |
| **CPU INFERENCE / live Anki** | **end-to-START** (`now() - last_review_time`) |

That is precisely the failure §9 exists to catch: each path is self-consistent in isolation, no
gate compares them, and the divergence is exactly `duration(k)`.

**It is sharper for us than for FSRS**, because our model already treats `duration(k)` as
unavailable: it is input feature 7, and the deploy contract *zeroes the most recent review's
duration* for that exact reason. We remove it from the features and then hand it back inside the
interval.

**=> The e2s arm is not a speculative experiment; it is the arm that MATCHES DEPLOY.** Run it.

---

## 9. ★★★ THE ROW-COUNT PROBLEM, AND WHY THE 0.0003 IS CONFOUNDED (Andrew, 2026-08-30)

Andrew: *"End-to-start results in a different number of reviews compared to end-to-end ... a
footnote is a good option as long as the number of reviews is exactly the same, but end-to-start
removes, on average, 0.172% of reviews because sub-1s reviews are filtered out (delta_t > 0, and
elapsed_seconds is integer seconds)."*

Confirmed in code. Under `--secs`, `features/base.py:227` reassigns `delta_t := delta_t_secs`, and
then `:284` returns `df[df["delta_t"] > 0]` (with a second zero-drop at `:271`). `elapsed_seconds`
is whole seconds, so subtracting a millisecond `duration` pushes sub-second gaps to 0 or below and
the filter deletes them. This is the section-6 flooring artifact, now with a consequence section 6
did not draw: **it changes the DENOMINATOR.** That is our own gate #1 — *equalized review count
IDENTICAL; any change is a pipeline bug* — applied to srs-benchmark.

### The dropped rows are NOT a neutral 0.178%

`scratchpad/.../dropped_rows.py`, 40 stride-sampled users, 2,179,548 real gaps:

| | value |
|---|---|
| dropped by `delta_t > 0` | 3,873 = **0.178%** of all rows |
| of those, same-day | **100.00%** |
| as a share of same-day rows | **0.458%** |
| their end-to-end gaps | median **2 s**, p90 15 s, max 60 s |
| failure rate, all rows | 16.18% |
| failure rate, **dropped** rows | **6.09%** |
| failure rate, kept same-day rows | 16.14% |

**The dropped rows are 2.7x EASIER than the rows that replace them in the denominator.** Deleting
the easiest 0.178% mechanically raises mean logloss for the end-to-start arm, with no help from the
interval definition at all.

### How big is that mechanical penalty? Possibly the entire result.

With `f = 0.00178` and `L_D` the mean logloss the model assigns the dropped rows,
`L_kept = (L - f*L_D) / (1 - f)`:

| L_D (dropped-row logloss) | apparent penalty to end-to-start |
|---|---|
| 0.25 | +0.00012 |
| 0.227 (marginal entropy at 6.09% failure) | +0.00016 |
| **0.15** | **+0.00030 = the entire observed gap** |

These are 2-second gaps with a 6% failure rate, so a loss far below the 0.3179 average is close to
certain and below 0.15 is entirely plausible. **=> The measured +0.0003 cannot be attributed to the
interval definition until the denominators match.** p = 10^-336.6 does not help here: a confound
this systematic is *more* consistent across users, not less.

**★ ONE CHEAP MEASUREMENT REPLACES THE WHOLE ESTIMATE, AND NEEDS NO RE-RUN.** From the END-TO-END
run's per-row predictions, take the mean logloss on exactly the rows end-to-start drops. That one
number turns the table above into a fact.

### The fix: floor the interval at 1 s (Andrew). This is right, and it is not a fudge.

`delta_t_secs = max(elapsed_seconds - duration/1000, 1) / 86400`

1. **Row count is preserved exactly.** Nothing can fall to 0, so `delta_t > 0` drops nothing and
   both columns score identical review sets.
2. **It invents nothing the data could have told us.** `elapsed_seconds` is integer seconds, so
   for these rows the true end-to-start gap is already only known to lie in `[0, 1)` s -- the
   physical inequality `start(k) >= answer(k-1)` bounds it below, the flooring bounds it above.
   1 s is the smallest value the column's own resolution can express. The error is under one
   second, which is the same bound the whole artifact already carries.
3. **It removes the confound above**, which is the real reason to do it.

Rejected alternatives: scoring the intersection of both arms (correct, but then neither column is
the standalone benchmark number), and a sub-second epsilon (invents precision the column does not
have, and parks 0.178% of rows at an identical fake interval).

**★ AND THIS PROMOTES PR #2 FROM CONVENIENT TO PREFERABLE.** With real millisecond timestamps a
0.4 s gap stays 0.4 s and survives `> 0` on its own merits -- no floor, no invented value, and only
genuine overlaps drop (2 in 2,306,229). The in-benchmark one-liner *cannot* reach that, because the
published column has no sub-second resolution. The 1 s floor is the correct fix for the data
everyone already has; the builder is the correct fix.

### ⚠ Does this hit OUR pipeline? No -- checked, not assumed.

* We have **no `delta_t > 0` filter.** `data_processing` keeps every row and only marks which ones
  count, via `label_is_equalize = label_review_th.isin(equalize_review_ths)`.
* The e2s tomls **reuse `label_filter_db`** and the same user ranges; they differ from their `_fix`
  twins by exactly one line (`LMDB_PATH`). The selected set is keyed by `review_th`, which is
  precomputed and identical.
* `elapsed_end_to_start_published` clamps to **0, not 1**, which is correct for us: we keep the row
  and give it a truthful ~0 elapsed, and `scale_elapsed_seconds` uses `log(1 + 1e-5 + x)`, so 0 is
  safe. The 1 s floor exists to survive a filter we do not have.

**So our featA2-vs-e2s A/B should satisfy gate #1 by construction -- and it is checked at eval,
not assumed**, since `size` is reported per user.

---

## 10. ★★★ RESULT WITH MATCHED SIZE: TWO THIRDS OF THE EFFECT WAS THE DENOMINATOR

Andrew re-ran both arms with intervals floored at 1 s, so `size` is identical:

| arm | size | LogLoss | AUC |
|---|---|---|---|
| e2e (unfloored) | 519,294,735 | 0.317944 | 0.752031 |
| e2s (unfloored) | 518,399,300 | 0.318275 | 0.751524 |
| **e2e (min1s)** | **519,486,445** | **0.317929** | **0.752117** |
| **e2s (min1s)** | **519,486,445** | **0.318040** | **0.751862** |

| | unfloored | matched size | shrank by |
|---|---|---|---|
| LogLoss gap | +0.000331 | **+0.000111** | **66.5%** |
| AUC gap | -0.000507 | **-0.000255** | **49.7%** |

**Section 9's confound was real and it was most of the result.** Solving
`penalty = f(L - L_D)/(1-f)` at `f = 0.001724` gives the dropped rows a mean logloss of
**~0.19** against the 0.318 average -- inside the 0.15-0.227 bracket section 9 predicted from
their 6.09% failure rate. The mechanism is confirmed quantitatively, not just directionally.

### ★ A SEPARATE FINDING, AND IT IS NOT ABOUT INTERVALS AT ALL

Flooring rescued **191,710 rows (+0.0369%) in the END-TO-END arm too**, which is why its numbers
moved. Those are reviews whose *answer-to-answer* gap floors to 0 s -- two answers on one card
inside the same second -- and `delta_t > 0` has been silently deleting them all along. **`min1s`
is therefore a fix worth making on its own merits, whatever is decided about the interval
definition.** Row totals reconcile exactly: 1,087,145 - 895,435 = 191,710.

### What survives is small, and it is coherent with the leak

+0.000111 LogLoss and -0.000255 AUC on identical rows. **AUC is the cleaner signature here**: it is
rank-based, so it measures discrimination, which is exactly what a leak buys and what calibration
metrics dilute. RMSE(bins) stays 6.36% in both arms.

**The magnitude fits the mechanism rather than embarrassing it.** `duration(k)` never reaches the
model as a feature -- it enters only as a perturbation of the interval, only on same-day rows
(38.8% of rows with a real gap), and even there it is a median **0.96%** of the gap. A ~1%
perturbation of one input transmits a sliver of a quantity whose standalone conditional AUC is
0.618. So a ~0.0003 AUC gain is what a weakly-coupled leak predicts; a large gain would have been
the surprise.

### The one check still open, now that it is finally clean

**Split the residual by same-day versus longer-than-a-day.** The two definitions are numerically
near-identical on long intervals, so a direct interval effect must live almost entirely in same-day
rows; anything appearing on long rows is the **refit** (same-day rows moved the fitted parameters,
which moved every prediction). Before the sizes matched, this test was contaminated by the drop --
the deleted rows were 100% same-day. Now it is a clean measurement.

### Recommendation, firmer than in section 8

The residual is the price of not using a quantity that does not exist at prediction time. That is
the **correction of an overestimate, not a cost.** At +0.0001 LogLoss, -0.0003 AUC and unchanged
RMSE, no ranking can move -- so a third table is harder to justify now than before, not easier.
Footnote, or replace. Hedging with a third table buys nothing at this effect size.

---

## 11. THE E2S DBS ARE VERIFIED IN THE DB, NOT JUST IN THE FUNCTION (2026-08-30)

`smoke_e2s_published.py` checks the transform as a FUNCTION. That is not the same as checking the
LMDB a training run will actually read. `scratchpad/features_rebuild/verify_e2s_columns.py` checks
the result, against a prediction written from the column list rather than from the output:

| | column | |
|---|---|---|
| must change | 2 `scaled_elapsed_seconds`, 3/4 its sin/cos, 5 the cumsum, 6/7 that sin/cos | all changed |
| must NOT change | 0 `scaled_elapsed_days`, 1 its cumsum, 8 `scaled_duration`, 9-23 the rest | **all byte-identical** |
| direction | e2s <= e2e always | **0 of 415,668 changed entries got larger** |

19.46% of `scaled_elapsed_seconds` entries moved, consistent with same-day rows being ~39% of the
set and roughly half of those carrying a material duration.

**Column 0 is the load-bearing one.** `is_first_review` IS `elapsed_days == -1`, so a transform that
touched the day index would re-label mid-card reviews as first reviews and poison the label
machinery. It did not move.

**The direction check is what a value-level diff cannot give you.** A transform that subtracted the
wrong duration, or subtracted it from the wrong side, would still "change the right columns" -- it
would just be wrong. Requiring monotone decrease tests the SEMANTICS, and it is free.

⚠ `note_id_is_nan` (13) is in this tensor and did not move, which is the expected result: the Bug C
difference between these two dbs lives in the id STREAMS, which are separate LMDB keys.
