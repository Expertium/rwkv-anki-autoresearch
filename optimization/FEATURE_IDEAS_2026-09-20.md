# New input features: the queued one, and what else is worth adding (2026-09-20)

**Andrew, 2026-09-20:** *"don't forget to do a run with one new input feature: total number of cards
reviewed at least once. Take a look at all the input features we have so far + that one, and see if
maybe there is some other new feature we could add ... try to be creative. Maybe there are some
interesting non-obvious things like your 'circular mean' of time of the day."*

The layout today is **103 inputs** (`INPUT_FEATURES.md`; 63 named columns, ratings one-hot to four).

---

## 0. The rebuild question, answered exactly -- and my first answer was wrong

`cum_new_cards` appears in `data_processing.py`'s `keep_columns`, and I first read that as "it is in the
database". It is not. `keep_columns` is a DATAFRAME-stage list; what the LMDB actually stores per chunk is
`card_features` (the model's input matrix), `global_labels`, `label_review_ths`, **`review_ths`**,
`day_offsets`, `day_offsets_first`, `skips` and the per-module ids. A column that is not in
`CARD_FEATURE_COLUMNS` is computed, used for other columns, and thrown away.

**So a new per-row feature normally needs a ~7 h rebuild AND a new `size` baseline.** Three exceptions,
and they are what makes the next section cheap:

1. **`review_ths` IS stored per row.** The user's absolute review index is therefore free, exactly, with
   no rebuild and no chunk-boundary problem (it is an absolute number, not a running total).
2. **A cumulative count can be reconstructed EXACTLY from a small side table**, because the quantity is a
   function of `(user, review_th)` and the row already carries `review_th`. For `cum_new_cards`: collect,
   per user, the sorted list of `review_th` values that are FIRST reviews (~5,061 per user at the median,
   ~100 MB for 7,500 users as int32), and then
   `cum_new_cards(row) = searchsorted(first_ths[user], review_th(row), side="right")`.
   One CPU pass over the `-id` parquet builds it; `review_th = parquet row + 1` is already verified in
   this repo (`scratchpad/r2/score_second.py`). Exact in train, eval and deploy, and immune to chunking.
3. **Anything derivable from a stored input column** is free: the first-review sentinel is a unique value
   of `scaled_elapsed_days`, deck identity is in the module ids, and so on.

⚠ **Why chunking is the thing to check for every candidate.** Training splits a user into a median of 4
chunks; eval uses 1. A feature computed as a running total FROM THE CHUNK START is therefore a different
quantity in training than at eval and deploy -- section 9's failure shape exactly. A candidate is cheap
only if it is absolute (route 1), or reconstructable from `review_th` (route 2), or local to a few rows.

---

## 1. Andrew's feature: total cards reviewed at least once

**Definition** (his): at each review, the number of DISTINCT cards this user has reviewed at least once,
counting the current card when this is its first review -- a scheduler knows it is showing a new card. An
upper bound on what the user could know.

* **Transform:** `scaled_cum_new_cards` = z-scored `log(1 + cum_new_cards)`, constants measured on the
  TRAIN half only (`optimization/feature_stats_id.py`).
* **Query rows: KEEP.** It describes the user, not the outcome.
* **Route:** 2 (the side table). No rebuild, no new `size` baseline, so it is gated against the current
  lineage directly.
* **Deploy:** one integer per collection, +1 on a card's first review. `DEPLOY_FUNCTIONS.md` section 4.
* **Cost:** ~1 CPU-hour to build the table, then one arm at the 1+1 budget (~5 h on the box) or at L*=1
  locally, against its twin.

---

## 2. Free or nearly free, and they should be measured beside it

| # | feature | why it might matter | route |
|---|---|---|---|
| F1 | **`review_th`** -- the user's total reviews so far | volume/experience; a 200-review beginner and a 200,000-review veteran are different populations, and nothing in the input says which | 1 (stored) |
| F2 | **this card's review number** | FSRS's `reps`. The RNN can count it, but an explicit counter frees capacity | 2 |
| F3 | **`cum_new_cards / review_th`** -- the share of activity spent on NEW cards | separates a cramming beginner from a mature collection in maintenance, in one dimension | 1+2 |
| F4 | **this card's share of the user's reviews** (`reps / review_th`) | how much of the user's attention this card has absorbed | 2 |
| F5 | **is this the user's first review today** | the start-of-day effect (cold start, no warm-up) -- `cum_reviews_today == 0`, already an input, but the model must learn the threshold | free |

---

## 3. The creative list -- non-obvious candidates, in the spirit of the circular mean

### A. SESSION STRUCTURE (my top pick after Andrew's)
The model sees the gap since any review, but nothing about WHERE in a study session the review sits. Cut
the history into sessions at a gap threshold (10 min, the same shape as the clock fix's T=1800) and add:
* **position in the session** (log-scaled), and **minutes since the session started**;
* **the session's length so far against the user's median session length**.

*Why non-obvious:* serial position within a session is a real memory effect (warm-up, then fatigue), and
it is invisible to the current inputs -- two reviews with the same 30-second gap look identical whether
they are the 3rd or the 300th of a sitting. *Route:* the gap is an input already; the counter resets at
session starts, which almost never cross a 65,536-row chunk boundary, and the rare crossing can be made
exact with the same `review_th` trick. *Risk:* the threshold is arbitrary; use two (5 min, 30 min).

### B. MORE CIRCULAR STATISTICS (extending `tod_dev`)
The current deviation is against the USER's circular mean time-of-day. Three relatives:
* **deviation from the CARD's own mean review time** -- a card always studied at breakfast, now shown at
  midnight, is being tested out of its encoding context;
* **deviation from the user's mean for THIS WEEKDAY** -- weekend schedules are genuinely different;
* **the user's phase coherence R** (the resultant length of their time-of-day distribution): a scalar
  saying how REGULAR the user is. An interval means something different for a metronome and for a binger,
  and this is the one number that tells them apart.

*Why non-obvious:* R is a property of the distribution, not of the review, and it modulates how much the
clock features should be trusted. *Route:* per-user and per-card running circular means -- deploy-safe
(the Anki builder already keeps running per-user statistics for `t_since_any_review`).

### C. SPACING DYNAMICS
* **interval growth ratio**: this gap / the card's previous gap. FSRS's stability ratio, which our model
  must currently infer from two separate rows.
* **overdue-ness**: this gap / the card's longest previous gap -- how far past its proven retention the
  card is being tested.

*Route:* per-card lag of one stored column; exact except on a card's first row inside a chunk.

### D. CONTEXT SWITCHING
* **same deck as the previous review?** and **same note as any review in this session?**
* Sibling interference beyond `scaled_sibling_gap`: the sibling's RATING, if a sibling was reviewed
  earlier in the same session (a just-failed sibling predicts this card's failure).

*Why non-obvious:* interference is a between-card effect, and every current feature is within-card.
*Route:* deck ids are stored; the sibling's rating needs a per-note running value (deploy-safe).

### E. RUNNING OUTCOME STATISTICS (the strongest, and the most expensive)
* **the user's running pass rate** (all-time, and over the last 50 reviews);
* **the card's lapse count** and **reviews since its last lapse**.

*Why:* these are FSRS's core state and our model must re-derive them through the recurrence every time.
*Route:* a running total of an OUTCOME, so it needs route 2 with a bigger side table, and it must use the
SHIFTED total (strictly before this review) or it leaks the label. **The shift is the whole risk here:
an off-by-one makes the feature the answer.** Worth doing carefully, after A.

### F. ANSWER-SPEED DYNAMICS (deploy-legal, and already half-built)
The current review's duration is zeroed by the deploy contract, but PAST durations are legal -- R1 built
running medians for exactly this (`rwkv/probe_impute.py`).
* **this card's previous duration against the user's median** -- a slow answer last time is a weak memory;
* **the user's recent speed against their all-time speed** -- a fatigue/state proxy for the whole session.

### G. CALENDAR BEHAVIOUR
* **study streak length** (consecutive days with at least one review) and **days missed in the last 30**:
  habit strength, which the current calendar features describe only as phase.

---

## 3a. ANDREW'S DECISIONS, 2026-09-20 -- this supersedes section 3's list

| idea | his call | what it becomes |
|---|---|---|
| A. Session structure | **YES**, asked how the cut is defined | measured, see below |
| B1. Phase coherence R | **YES**, asked for the math | see below |
| B2. Card-level circular mean | **NO** -- dropped | -- |
| C. Interval growth ratio | **NO** -- *"scheduler-dependent, so not a property of memory (unless intervals were given by a perfect oracle scheduler, which they very much weren't)"* | dropped, and the same objection retires **overdue-ness**, which is the same ratio against the card's longest gap |
| D. Sibling interference (the sibling's rating) | **YES** | -- |
| E. Running pass rate | **REPLACED** -- *"that looses signal, so let's use lossless version: running mean of grades. That way RWKV can distinguish between 'on average the user presses Hard' and 'on average the user presses Easy'"* | running mean of the RATING (1-4), not of pass/fail |
| G. Study streak | **YES** | -- |

**⚠ C's objection generalises, and it is the sharpest methodological point in this list.** An input that is a
function of the SCHEDULER's choices describes the policy that produced the data, not the memory we are modelling.
Our own interval features survive only because the interval is also the elapsed time the forgetting curve is
evaluated at -- a physical quantity. A RATIO of two scheduler-chosen intervals has no such reading.

### A. The session cut -- measured, not chosen (`scratchpad/sessions/gap_antimode.py`)

Andrew asked *"how will we cut sessions? >2h gap between reviews or something like that?"*. The non-arbitrary
answer is the ANTIMODE of the gap distribution, so it was measured: 40 VAL users, 4.8 M consecutive-review gaps
from `review_time`, 0.1-decade log bins.

**The honest result is that there is no sharp cut.** The within-sitting mode is at ~4 s (9.3% of gaps in one bin);
density then falls monotonically to a long FLAT SHOULDER of 0.04-0.06% per bin running from ~15 min to ~6 h, and
the between-day mode sits at ~1 day (0.16%). The computed antimode is 6.2 h, but it is only marginally emptier
than the whole shoulder -- i.e. the trough is a plateau, and any threshold inside it is defensible.

| threshold | gaps that break a session | mean session |
|---|---|---|
| 1 min | 10.85% | 9.2 reviews |
| 5 min | 2.63% | 38.1 |
| 10 min | 1.89% | 52.9 |
| **30 min** | **1.37%** | **72.9** |
| 1 h | 1.18% | 84.9 |
| 2 h (Andrew's guess) | 1.01% | 99.1 |
| 6 h | 0.77% | 129.8 |

**=> the choice between 5 min and 6 h moves the session count by 3.4x and the mean session length from 38 to 130
reviews, and the histogram does not pick for us.** So: (1) the primary cut is **30 min**, because the pipeline
ALREADY has that constant -- the clock fix's `T=1800` is our definition of "the same sitting" -- and one word for
one thing beats a second, differently-drawn boundary; (2) ship the threshold-FREE forms beside it, which carry the
same signal without the free parameter: **the count of reviews in the trailing 10 minutes and in the trailing 60
minutes**. A rolling count is monotone, needs no session concept, and degrades gracefully where a threshold snaps.
(3) 2 h is not wrong -- it is the far end of the same plateau, and it merges two sittings in one evening, which is
the fatigue signal we are after; if the 30-min arm pays, 2 h is the cheap variant to try second.

**⚠ Do not put a threshold near 60 s.** The 50-65 s bin holds 5.5% of all gaps against 2.5% and 1.2% in the bins
either side -- a spike, and partly an artefact: the later review is in learning/relearning 51.4% of the time there
against 40.5% in the neighbouring 20-45 s band, i.e. the scheduler's 1-minute learning step explains some of the
pile-up and user behaviour the rest. A cut drawn through it would key on the scheduler.

### STATUS 2026-09-21 -- BUILT, NOT QUEUED

Everything Andrew kept except the sibling rating is implemented and verified behind
`RWKV_EXTRA_FEATURES` (`rwkv/extra_features.py`), on a per-ROW side table (`scratchpad/rowtables/`):
`sess_pos`, `sess_secs`, `roll10`, `roll60`, `phase_r`, `phase_r2`, `streak`, `grade_mean` -- eight
columns, 111 inputs with all of them on. Checks: `scratchpad/rowtables/smoke_rowfeat.py` (alignment
against the LMDB's own `tod_sin`/`tod_cos` and `cyc30`/`cyc100`; a table shifted by one row fails 9
of 9 chunks), `deploy_vs_lmdb.py` (0 columns disagree, real and query rows), the 111 width case, and
`leak_check.py` for `grade_mean` (**PASS**, see below). Pre-registration:
`scratchpad/rowtables/PREREG.md` -- ONE arm, leaning null, and gated on what `newc1` says first.

The SIBLING RATING is not built. It needs a third table (note membership) and it is the one kept
item that is structurally different from the rest -- it is a cross-entity fact rather than an
accumulation -- so it is held until `newc1` says whether the accumulation route pays at all.

**The visualization Andrew asked for:** `scratchpad/phase_coherence/phase_coherence.png`, drawn from
three real users rather than from the formula (user 330 R=0.94 a metronome; user 138 R=0.03/R2=0.03
genuinely spread; user 229 R=0.05 but R2=0.46, two sittings 12 h apart -- the case R alone calls
random). Median whole-history R over 400 scanned users is 0.427.

**The leak check took three verdict rules** and only the third was worth anything: a correlation bar
mislabels the honest 0.42 at a one-review history; an exact-recovery bar mislabels constant users
whose past mean equals the button they press; the answer is a within-user permutation floor, where
the shipped column sits 1.5x above its floor and the deliberately unshifted copy sits at 100%.

### B1. Phase coherence R -- the math

Each review has a time-of-day angle on the 24 h circle:

    theta_i = 2*pi * ((review_time_i mod 86400) / 86400)

Over the user's first n reviews keep two running sums (this is all the state deploy needs):

    C_n = sum_i cos(theta_i)        S_n = sum_i sin(theta_i)

Then

    R_n = sqrt(C_n^2 + S_n^2) / n            in [0, 1]      <- the resultant length = THE FEATURE
    mu_n = atan2(S_n, C_n)                                  <- the circular mean, already our `tod_dev` base

`R = 1` means every review at the same clock time (a metronome); `R = 0` means uniform over the day (or, degenerately,
two exactly opposite peaks -- see below). The related forms are circular variance `1 - R` and circular standard
deviation `sqrt(-2 * ln R)`; R itself is the well-conditioned one for an input because it is bounded.

**Three details that decide whether it works:**

1. **R is biased upward at small n.** For a genuinely uniform user, `E[R^2] ~ 1/n`, so a 9-review user reads
   `R ~ 0.33` and looks regular. Use the standard bias correction, which is Rayleigh's statistic rearranged:
   `R2_unbiased = (n*R^2 - 1) / (n - 1)`, feeding `sqrt(max(R2_unbiased, 0))`. Feed `log1p(n)` alongside so the
   model can discount what is left. Without this the feature is mostly a review-count proxy at the low end.
2. **Use the running value strictly BEFORE this review**, like every other counter here: add `theta_i` to `C`/`S`
   only after the row is emitted. Not a label leak (no outcome is involved), but it keeps train, eval and deploy
   computing one quantity, and it is what a scheduler can compute.
3. **Bimodality is invisible to R.** A user who studies at 08:00 and 20:00 is perfectly regular and scores `R ~ 0`,
   the same as a random user. The cheap fix is the SECOND circular moment, one more pair of accumulators:
   `R2 = |sum exp(2i*theta)| / n`, which is high for a two-peaked user and low for a uniform one. Two dims total.

**Deploy cost:** 4 floats and an integer per collection (`C`, `S`, `C2`, `S2`, `n`), updated with two cosines and
two sines per review. Chunk-immune by construction if the accumulators are seeded from the user's whole history --
which, unlike a session counter, means training must NOT restart them at a chunk boundary: they are reconstructed
from `review_th` the same way `cum_new_cards` is, or carried in a per-user side table of prefix sums.

### E. Running mean of grades (replaces the pass rate)

    grade_mean_before(row) = mean(rating of this user's reviews strictly before this one)      in [1, 4]

and the same per card. Andrew's reason is exactly right and worth keeping: pass/fail collapses Hard-vs-Good and
Good-vs-Easy, which is the very distinction the rating head predicts, so the pass rate throws away the part of the
signal our weaker mode is graded on. Scale to `(mean - 2.5) / 0.75` or z-score on the train half.
**This is the one candidate that can leak**, because it is a running mean of the LABEL family: an off-by-one makes
`grade_mean` include this review's own rating and the imm head reads the answer. The check is pre-registered:
build the feature, then build a deliberately UNSHIFTED copy, and require the unshifted one to be near-perfectly
predictive of `label_rating` while the shifted one is not. A leak test that cannot fire proves nothing.

## 4. What I recommend, in order

*(updated 2026-09-20 to Andrew's decisions; the dropped items are gone from the order.)*

1. **Andrew's `cum_new_cards`, alone, as one arm** -- it is his ask, it is now rebuild-free, and it is the
   control for everything below.
2. **F1 + F2 + F3 beside it as a second arm** (the free counters). Three dims, one run.
3. **Session structure (A)** at the 30-min cut, plus the two rolling counts -- the most promising genuinely new
   signal, and the rolling counts make it robust to the threshold the histogram refused to pick.
4. **Phase coherence R + the second circular moment (B1)** -- two dims, four accumulators.
5. **Running mean of grades (E, Andrew's lossless form) + the card's lapse count + study streak (G)** -- highest
   expected value, and the only leak risk in the set; build the shift check FIRST (the same shape as the KD test:
   with the shift removed the feature must become near-perfectly predictive of `label_rating`, which is how you
   prove the shifted one is honest).
6. **Sibling interference (D)** -- the sibling's rating plus a "no sibling this session" flag, 2 dims.

**Dropped by Andrew:** the card-level circular mean (B2), the interval growth ratio and overdue-ness (C, both
scheduler-dependent), and the pass rate as such (replaced by the grade mean).

Everything above is one dimension each except A (three) and B (three). The trunk is 561,732 parameters and
the input projection is 103 -> 320, so each new dim costs 320 parameters, i.e. rounding error.

⚠ **Gate note:** these arms change the INPUT LAYOUT but not the `label_filter_db`, so `size` is unmoved
and they are gated against the current lineage with the usual both-modes rule. The layout change means a
new Rust export and a fresh parity trace before anything ships, which is mechanical.
