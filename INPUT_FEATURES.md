# RWKV input features — the full 92-dim per-review vector

Source of truth: `rwkv/data_processing.py` (dense features + query masking),
`rwkv/prepare_batch.py` (ID + day-offset encodings, `add_encodings`),
`rwkv/config.py` (dims + periods). One conceptual feature per row; sin/cos pairs and
multi-dim encodings are counted as one feature (the `dims` column sums to **92**).
"log-z" = `(log(...) − mean)/std` with the constants in `STATISTICS`
(`data_processing.py`).

| # | Feature | Dims | What it is | Transform / encoding |
|---|---|---|---|---|
| 1 | Elapsed days | 1 | Days since **this card's** previous review (the interval length) | `log(1+1e-5+x)` log-z; first review (−1) → 0 |
| 2 | Elapsed days, cumulative | 1 | Running sum of the card's elapsed_days (card "calendar age") | log-z |
| 3 | Elapsed seconds | 1 | Seconds since this card's previous review | log-z |
| 4 | Elapsed-seconds sub-day phase | 2 | Where the interval falls within a 24 h day (distinguishes e.g. a 1.0-day from a 1.5-day gap; absolute time-of-day is unrecoverable in the anonymized data) | `sin`, `cos` of `2π·(elapsed_seconds mod 86400)/86400` |
| 5 | Elapsed seconds, cumulative | 1 | Running sum of the card's elapsed_seconds | log-z |
| 6 | Cumulative-seconds sub-day phase | 2 | 24 h phase of the cumulative clock | `sin`, `cos` |
| 7 | Review duration | 1 | Answer time of this review | `log(10+x)` log-z |
| 8 | Grade | 4 | The rating given: Again / Hard / Good / Easy | one-hot |
| 9 | Missing-ID flags | 3 | Note / deck / preset ID was missing (missing notes get a unique per-card placeholder; missing decks/presets share one placeholder) | 0/1 each |
| 10 | Days since any review | 1 | Days since the user's previous review of **any** card | `log(log(e+x))` |
| 11 | Pseudo-day-of-week | 1 | Position in a 7-day cycle (phase arbitrary — day 0 is anonymized) | `((day_offset mod 7) − 3)/3` ∈ [−1,+1] |
| 12 | New cards since card's last review | 1 | How many **new** cards the user introduced between this card's previous review and now | `log(3+x)` log-z |
| 13 | Reviews since card's last review | 1 | How many **other** reviews the user did in that same window | `log(3+x)` log-z |
| 14 | New cards today | 1 | Running count of new cards introduced so far today | `log(3+x)` log-z |
| 15 | Reviews today | 1 | Running count of reviews done so far today | `log(3+x)` log-z |
| 16 | Card state | 1 | Anki card state (new/learning/review/relearning) | `state − 2` |
| 17 | Query flag | 1 | 1 on the synthetic "predict cold" rows used by ahead mode (see masking note below) | 0/1 |
| 18 | Card ID | 12 | Identity of this exact card | random code per entity, each dim uniform over {−1.5,−0.5,+0.5,+1.5}; **re-randomized every batch** (see note) |
| 19 | Sibling (note) ID | 12 | Identity of the note — cards generated from the same note share it | 〃 |
| 20 | Deck ID | 8 | Identity of the deck | 〃 |
| 21 | Preset ID | 8 | Identity of the deck-options preset | 〃 |
| 22 | 3-day cycle | 4 | Position of the review day in a 3-day cycle, plus the same for the day this card was **first** reviewed (card-cohort anchor) | `sin`, `cos` × {review day, first-review day}; random per-batch phase baseline |
| 23 | Pseudo-week cycle (7 d) | 4 | 〃 for a 7-day period | 〃 |
| 24 | Pseudo-month cycle (30 d) | 4 | 〃 for a 30-day period | 〃 |
| 25 | Pseudo-quarter cycle (100 d) | 4 | 〃 for a 100-day period | 〃 |
| 26 | Pseudo-year cycle (365 d) | 4 | 〃 for a 365-day period | 〃 |
| 27 | Pseudo-decade cycle (3650 d) | 4 | 〃 for a 3650-day period | 〃 |
| 28 | Pseudo-century cycle (36500 d) | 4 | 〃 for a 36500-day period | 〃 |
| | **Total** | **92** | | |

## Notes

- **Query masking (ahead mode):** each real review row gets a paired row with
  `is_query = 1` on which every answer-derived column is zeroed via the explicit
  keep/reject lists in `add_queries` (`data_processing.py`): the grade one-hot,
  duration, and card state are rejected; all timing, ID, and counter features are
  kept. So ahead-mode predictions see interval/context information only.
- **ID codes are NOT learned embeddings** — they are re-drawn randomly every batch
  (`randint(0, ID_SPLIT=4) − 1.5` per dim). Identity is carried purely by code
  *matching* within the sequence ("same code as an earlier review" = same
  card/note/deck/preset). `user_id` gets no code (a sequence is always one user).
  The same IDs also route each review into the 5 chained RWKV streams
  (card → note → deck → preset → user), so identity enters the model twice: as these
  input codes and as the per-entity recurrent-state partitioning.
- **Cycle features** (rows 22–28, `DAY_OFFSET_ENCODE_PERIODS` in `config.py`): the
  phase `baseline` is a random integer in `[0, P)` drawn per batch — augmentation so
  the net can't memorize absolute positions in a cycle, only relative structure.
- **Labels** (training targets, not inputs; from the card's *next* review):
  `label_y`, `label_rating`, `label_elapsed_days`, `label_elapsed_seconds` — the
  forgetting-curve head is supervised at the actual next-interval point;
  `label_is_equalize` marks reviews that count in the benchmark.
- ⚠ **Invariant** (optimization protocol): the model must keep running on this exact
  92-dim preprocessed input / the existing LMDBs — no new or changed inputs.
