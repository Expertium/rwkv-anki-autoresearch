# Are the input features used correctly, and is anything being thrown away?

Andrew, 2026-09-09, while arm 1 holds the GPU: *"check data pre-processing to make sure all input
features (especially new ones) are used correctly and that we aren't accidentally throwing away
useful info"*.

Method: read the code, then **read the gen-5 LMDB back** and report what the model actually
receives. A column can be derived correctly and then be dropped, mis-ordered, degenerate, or coded
so its "undefined" case is indistinguishable from a real value -- and only the stored tensor shows
that. Tools in this directory (`audit_columns.py`, `probe_two.py`, `nan_rate_ab.py`,
`dupe_scan.py`, `sentinel_check.py`, `card_coverage.py`, `raw_cols.py`); all CPU, seconds each,
zero GPU.

Samples: stride over users 1..4999 (step 157 / 311), first chunk each, real rows only unless said.

---

## Verified correct -- nothing to do

**1. No source column is silently discarded.** The `-id` dataset has exactly these raw columns:
`revlogs` = review_time, card_id, day_offset, rating, state, duration, elapsed_days,
elapsed_seconds; `cards` = card_id, note_id, deck_id; `decks` = deck_id, parent_id, preset_id.
Every one is consumed. The two that never reach the input vector are deliberate: **`state`**
(Andrew's rebuild directive) and **`parent_id`** (it derives `scaled_deck_depth`, and fed the
rejected deck-tree lever).

And **dropping `state` is a no-op, not an unpriced change** -- checked rather than assumed. 108
runners in `scratchpad/`, the published-lineage champion `run_iter53.cmd` among them, set
`RWKV_ZERO_FEATURES=22`, which is `scaled_state`. It has been masked out of the input for the whole
5k phase, so gen 5 removing the column changes nothing.

**2. All 69 card features reach the model.** Stored width 69, plus 40 ID codes = 109, and
`w10_ws_109350.pth`'s `features2card.0.weight` is `(320, 109)`. The width contract holds end to end.

**3. A new column cannot be silently omitted.** `add_queries` partitions every frame column into
keep/reject and asserts the partition is total (`data_processing.py:567-578`). Adding a column
without listing it fails the build rather than dropping it.

**4. No leak found in the new columns.** Each is a function of strictly prior rows or of the
current review's own clock, which the scheduler knows: the time-of-day deviation uses **exclusive**
prefix sums; the creation-batch counts are clipped at `review_time`; the sibling gap is restricted
to **preceding** siblings; `card_predates_first_review` compares against the user's first review.
All are correctly `keep_columns` -- they survive on the synthetic query row -- because none carries
the outcome.

**5. The `-id` build does NOT lose metadata relative to the published build.** 32 users, 261,074
rows, first chunk each: `note_id_is_nan` is **34.84% in both**, and the paired per-user difference
is **+0.000 pp, max 0.00** -- the same rows, not merely the same rate. The metadata is missing at
the source: for user 101 the cards table holds 10,360 cards while only 48.4% of the reviewed card
ids appear in it (a deleted card keeps its revlog rows and loses its card row), and the published
set -- whose ids are factorized independently -- reproduces the same 33.15% row-match exactly. So
the raw-id join is sound and this is a dataset property, not a pipeline defect.

Its consequence is worth stating plainly: on ~35% of training rows the note stream degenerates to
one note per card and the deck and preset streams collapse to a single synthetic entity, i.e. two
of the five streams carry nothing the `user_id` stream does not already have. That is the correct
handling of absent data, not a bug, but it caps what the deck and preset scopes can contribute.

---

## Coding warts -- real, none an outright loss, all needing a rebuild to fix

### W1. `scaled_sibling_gap`: the undefined case is coded as a legal value, with no flag

**97.37% of real rows are undefined** and are written as exactly `0.0`, which sits inside the
defined range `[-2.219, +1.508]`. `0.0` de-standardizes to a log-gap of 9.3354, i.e. **the model
reads "this note has no other card reviewed earlier" as "a sibling was reviewed about 3.1 hours
ago"**. Nothing marks the difference -- the best existing proxy is `cyc36500_cos` at |r| = 0.40.

The contrast inside the same file is the point. `scaled_deck_age_at_review` uses the **identical**
sentinel-as-0.0 coding and is fine, because `deck_id_is_nan` flags exactly when it is undefined --
`id_features.py` says so in as many words ("without it the net cannot tell 'no deck age' from 'deck
age 0'"). The sibling column never got its flag.

Severity, stated honestly: at 97% the sentinel is nearly a constant, so the input FC absorbs most of
it as a bias. The genuine cost falls on the 2.6% of **defined** rows whose true value lands near
0.0, which are read as undefined. Small -- and the LOO sweep already prices the column at
+0.000205 / +0.000487, so it works despite this.

Also: coverage is **2.6%**, against the **10-16%** CLAUDE.md pre-registered. That ceiling counted
"rows whose note had another card **created** earlier"; the shipped column needs a sibling
**reviewed** earlier, which is far stricter. The pre-registration used the looser definition.

**Fix at the next rebuild:** add `sibling_gap_is_defined`, or move the sentinel outside the range.

### W2. `note_id_is_nan`, `deck_id_is_nan`, `preset_id_is_nan` are the same column

Bit-identical on **100.000%** of rows, and mechanically so: deck comes from the card row and preset
from the deck row, so one failed card join sets all three. Two of the three input dims are exact
duplicates. Harmless; free to reclaim.

### W3. Two normalization constants are stale, and one for a knowable reason

The constants were measured **2026-08-03**; `t_since_any_review` became **end-to-start** on
2026-08-19 (the previous review's duration is now subtracted) and the constant was never
re-derived. Measured on the gen-5 db, where each column should be ~N(0,1):

| column | mean | std | min |
|---|---|---|---|
| `scaled_t_since_any_review` | **-1.108** | 1.584 | -1.664 |
| `scaled_user_tenure` | **-1.438** | 1.561 | **-15.812** |
| `scaled_creation_batch_1h` | -0.516 | 0.810 | -1.180 |
| `scaled_deck_age_at_review` | -0.332 | 0.971 | -2.594 |
| `scaled_sibling_gap` | -0.014 | 0.170 | -2.219 |

An affine error is absorbed by the input FC, so this costs conditioning rather than information --
which is why the file's own block comment tolerates up to 24% drift on the older constants. Two
things are still worth noting. `scaled_user_tenure` puts every user's **first** review at
**-15.8 sigma**, an outlier no other column produces. And ~78% of `scaled_t_since_any_review`'s
mass sits in three adjacent bf16 levels at its floor -- which is CORRECT, because end-to-start
measures the pause between answering one card and the next being shown, and that is milliseconds
inside a session; the informative part of the column is entirely its session-boundary tail.

### W4. `is_default_deck` is dead

**100.0%** zero in the sample. It exists to separate "no deck age" from "deck age 0" -- a job
`deck_id_is_nan` already does -- and Anki's default deck id (1) essentially never survives to a
reviewed row. One dim.

### W5. The long-period real cycles are near-duplicates of each other

`cyc3650_sin` vs `cyc3650_first_sin` **r = 0.9935**; `cyc36500_sin` vs `cyc36500_first_sin`
**r = 0.9910**; `cyc3650_cos` vs `cyc36500_cos` **r = -0.963**. `cyc36500_cos` has std **0.011**
over the range [-1.00, -0.83], i.e. it is nearly constant.

Expected, and it explains part of realcyc's exact tie: over one user's history the review day and
the card's creation day sit at almost the same phase of a decade or century cycle, and a century
"cycle" over this dataset is a linear ramp of the epoch day. So 8 dims (decade + century, review
and first halves) carry roughly 2 independent numbers. Not harmful -- the ID-code block they
replaced spent 28 dims on the same job -- but the cheap trim, if the layout is ever revised, is the
`_first` half of the two longest periods.

### W6. `scaled_deck_depth` folds "unknown deck" onto "top-level deck"

82.2% of rows sit at `_std(0)`; a missing deck and a genuine depth-0 deck take the same value.
Recoverable by the model through `deck_id_is_nan`, so this is a note, not a defect.

---

## What I did NOT change

Every fix above is a **build-time** change, so it costs an LMDB rebuild (~7 h) plus a re-base of the
gen-5 lineage. The endgame is running on gen 5 and arm 1 is the control every decay branch is gated
against, so none of this is worth interrupting it for. Recorded for the next rebuild, if there is
one.
