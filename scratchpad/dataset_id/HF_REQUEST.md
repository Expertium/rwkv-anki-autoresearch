Written by Claude

## Offering an `-id` variant: same tables, real Anki IDs + a `review_time` column

I have built and been using a variant of this dataset that keeps the raw Anki IDs instead of
factorizing them, and adds an absolute `review_time`. It unblocks a family of features that the
current parquets make impossible to even test, and I would like to contribute it — either as an
upload, or just as the builder diff if you would rather generate it yourselves.

**Important: this exposes nothing new.** It is built from your own
`open-spaced-repetition/anki-revlogs-10k-raw` (`revlogs.7z`, 8,459,427,959 bytes), with a builder
adapted from `anki-revlogs-dataset-builder/build_parquet.py`. Every value in it is already
decodable from the protobufs you host there; this is a decoded, analysis-ready view, not a new
disclosure. It should presumably inherit whatever access terms you consider right — happy for it
to be gated exactly like `anki-revlogs-10k`.

### What changes

Same three tables, same partitioning, same 1:1 user numbering, so it is a drop-in source.

| table | published | `-id` variant |
|---|---|---|
| `revlogs` | `card_id, day_offset, rating, state, duration, elapsed_days, elapsed_seconds` | **+ `review_time`** (epoch ms) |
| `cards` | `card_id, note_id, deck_id` | same columns, values are **raw epoch-ms IDs** |
| `decks` | `deck_id, parent_id, preset_id` | same columns, values are **raw epoch-ms IDs** |

Because Anki IDs *are* creation timestamps, keeping them raw is what makes the rest work.
Example (user 1, first row): `card_id = 1621696450781` → 2021-05-22 15:14:10 UTC, and its first
review `review_time = 1621697507001` → 15:31:47 UTC. So "card created 17 minutes before its first
review" reads directly off the data.

10,000 user directories in `revlogs` and `decks`, 9,934 in `cards`; ~16 GB of parquet.

### Why it is worth having

None of these are derivable from the published parquets, and several are plausible predictors that
currently nobody can test:

- time of day, and a user's deviation from their own typical study hour
- true calendar phase — real day-of-week and day-of-year, rather than an integer day counter
- card creation → first review gap
- seconds-resolution time since the user's *last review of anything*, not just of this card
- creation-batch size (how many cards were added in the same minute / hour / day)
- user tenure, and note / deck / preset age at the time of a review

### One correction worth upstreaming regardless

`review_time` is **not** `revlog.id`. Anki writes the revlog row when the user *answers*, so
`revlog.id` is the answer time. For elapsed-time and time-of-day features you want the moment the
card was *shown*:

```python
review_time = entry.id - entry.taken_millis
```

Verified in the data rather than assumed. User 333, 296,002 rows, 18,149 cards with ≥2 reviews:
show times are monotone per card on **all 18,149**, answer times (`review_time + duration`) on only
**18,008**. That asymmetry can only arise if the stored column is the show time and durations vary.
Everything downstream (`day_offset`, `elapsed_days`, `elapsed_seconds`, sort order) is recomputed
from the corrected value.

**Consequence a consumer should know about:** `elapsed_seconds` is diffed in protobuf order
(per-card blocks) and the frame is sorted by `review_time` only afterwards, so the show-time
correction can reorder two adjacent reviews of one card and leave a genuinely **negative**
`elapsed_seconds`. On user 333 that is **127 of 296,002 rows (0.043 %)**, ranging −58 s to −2 s
(median −55 s) — bounded by the review's own duration, exactly as the mechanism predicts. The
published set has none.

⚠ Count these as `elapsed_seconds < -1`, not `< 0`: `-1` is the existing "no known previous review"
sentinel and accounts for a further 6.15 % of rows in both versions, so a naive `< 0` test reports
6.19 % and hides the real figure.

Clamping the real negatives to 0 is the right fix — the overlap is bounded by the review's own
duration, so the true gap really is ~0. Clamping to the `-1` sentinel instead is **not** safe: it
propagates into any per-card cumulative sum of `elapsed_seconds` and turns into `log(negative)`
downstream. This is inherited from the upstream formula, so it is worth documenting either way.

### It is a faithful superset

Compared against the published set over 6 users (1, 2, 3, 17, 101, 555 — 363,598 reviews): row
counts identical user-for-user, and `day_offset` differs on **4 of 363,598 rows (0.001 %)**, all of
them reviews the show-time correction moved across a day rollover.

### What would you like?

1. I upload the built parquets (~16 GB) to a new repo under the org, or as a revision/subfolder
   here — whichever fits your layout; or
2. I contribute the builder diff against `anki-revlogs-dataset-builder` and you generate it.

Either is fine. (2) is less to host and keeps you in control of the artifact; (1) saves everyone
the ~38 GB staging extract and the CPU. Happy to do the work either way.
