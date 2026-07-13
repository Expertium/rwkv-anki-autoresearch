# RWKV input features — the full 92-dim per-review vector

Source of truth: `rwkv/data_processing.py` (dense features + query masking),
`rwkv/prepare_batch.py` (ID + day-offset encodings, assembled in `add_encodings`),
`rwkv/config.py` (dims + periods). The model input is **92 dims per review**:

| Group | Dims | Built in |
|---|---|---|
| A. Dense per-review features (`CARD_FEATURE_COLUMNS`) | 24 | `data_processing.py` |
| B. Random ID encodings (card/note/deck/preset) | 40 | `prepare_batch.py` |
| C. Cyclical day-offset encodings (7 periods × 4) | 28 | `prepare_batch.py` |
| **Total** | **92** | |

Z-scoring constants live in `STATISTICS` (`data_processing.py`); "log-z" below means
`(log(...) − mean) / std` with those constants.

## A. Dense per-review features (24)

In `CARD_FEATURE_COLUMNS` order:

| # | Feature | What it is | Transform |
|---|---|---|---|
| 1 | `scaled_elapsed_days` | Days since **this card's** previous review (interval length) | `log(1+1e-5+x)` log-z; first review (−1) → 0 |
| 2 | `scaled_elapsed_days_cumulative` | Running sum of the card's elapsed_days (card "calendar age") | log-z |
| 3 | `scaled_elapsed_seconds` | Seconds since this card's previous review | log-z |
| 4 | `elapsed_seconds_sin` | Sub-day phase of the interval: `sin(2π · (elapsed_seconds mod 86400)/86400)` | — |
| 5 | `elapsed_seconds_cos` | Cos of the same phase (together they distinguish e.g. 1.0-day vs 1.5-day gaps; absolute time-of-day is unrecoverable in the anonymized data) | — |
| 6 | `scaled_elapsed_seconds_cumulative` | Running sum of the card's elapsed_seconds | log-z |
| 7 | `elapsed_seconds_cumulative_sin` | 24 h phase of the cumulative clock | — |
| 8 | `elapsed_seconds_cumulative_cos` | 〃 | — |
| 9 | `scaled_duration` | Answer time of this review | `log(10+x)` log-z |
| 10 | `rating_1` | Grade one-hot: Again | 0/1 |
| 11 | `rating_2` | Grade one-hot: Hard | 0/1 |
| 12 | `rating_3` | Grade one-hot: Good | 0/1 |
| 13 | `rating_4` | Grade one-hot: Easy | 0/1 |
| 14 | `note_id_is_nan` | Note ID was missing (filled with a unique per-card placeholder) | 0/1 |
| 15 | `deck_id_is_nan` | Deck ID was missing (all such cards share one placeholder deck) | 0/1 |
| 16 | `preset_id_is_nan` | Preset ID was missing (shared placeholder) | 0/1 |
| 17 | `day_offset_diff` | Days since the user's previous review of **any** card | `log(log(e+x))` |
| 18 | `day_of_week` | Pseudo-weekday sawtooth `((day_offset mod 7) − 3)/3` ∈ [−1,+1] (phase arbitrary — day 0 is anonymized) | — |
| 19 | `diff_new_cards` | How many **new** cards the user introduced between this card's previous review and now | `log(3+x)` log-z |
| 20 | `diff_reviews` | How many **other** reviews the user did in that same window | `log(3+x)` log-z |
| 21 | `cum_new_cards_today` | Running count of new cards introduced so far today | `log(3+x)` log-z |
| 22 | `cum_reviews_today` | Running count of reviews done so far today | `log(3+x)` log-z |
| 23 | `scaled_state` | Anki card state (new/learning/review/relearning) | `state − 2` |
| 24 | `is_query` | 1 on the synthetic "predict cold" rows used by ahead mode | 0/1 |

**Query masking (ahead mode):** each real review row gets a paired query row with
`is_query = 1` on which every answer-derived column is zeroed via the explicit
keep/reject lists in `add_queries` (`data_processing.py`): the grade one-hots,
`duration`, and `state` are rejected; all timing, ID, and counter features are kept.
So ahead-mode predictions see interval/context information only.

## B. Random ID encodings (40)

Dims per `ID_ENCODE_DIMS` (`config.py`): `card_id` 12, `note_id` 12, `deck_id` 8,
`preset_id` 8. The **note ID is the "sibling ID"** — cards generated from the same
note share it.

| Property | Value |
|---|---|
| Code per unique entity | random vector, each dim uniform over {−1.5, −0.5, +0.5, +1.5} (`randint(0, ID_SPLIT=4) − 1.5`) |
| Lifetime | **re-randomized every batch** — these are NOT learned embeddings |
| What identity means to the net | code *matching* within the sequence ("same code as an earlier review" = same card/note/deck/preset) |
| `user_id` | no encoding (a sequence is always exactly one user) |

The same IDs also route each review into the 5 chained RWKV streams
(card → note → deck → preset → user), so identity enters the model twice: as these
input codes and as the per-entity recurrent state partitioning.

## C. Cyclical day-offset encodings (28)

For each period **P ∈ {3, 7, 30, 100, 365, 3650, 36500} days**
(`DAY_OFFSET_ENCODE_PERIODS`, `config.py`) — pseudo 3-day / week / month / quarter-ish /
year / decade / century cycles — four dims:

| Dims | Encoding |
|---|---|
| 2 | `sin`, `cos` of `2π · ((baseline + day_offset) mod P)/P` — where the review sits in the cycle |
| 2 | `sin`, `cos` of the same for `day_offset_first` — the day this card was **first** reviewed (card-cohort anchor) |

`baseline` is a random integer in `[0, P)` drawn **per batch** (augmentation: the net
can't memorize absolute positions in the cycle, only relative structure).
7 periods × 4 = 28 dims.

## Not inputs, but adjacent

- **Labels** (training targets, from the card's *next* review): `label_y`,
  `label_rating`, `label_elapsed_days`, `label_elapsed_seconds` — the forgetting-curve
  head is supervised at the actual next-interval point; `label_is_equalize` marks
  reviews that count in the benchmark.
- **Routing metadata** carried alongside the features (not concatenated into the 92):
  raw `card/note/deck/preset` IDs for stream routing, `day_offset`,
  `day_offset_first`, `review_th`, `has_label`, `skip`.

⚠ Invariant (optimization protocol): the model must keep running on this exact
92-dim preprocessed input / the existing LMDBs — no new or changed inputs.
