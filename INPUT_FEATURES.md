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

## Simplified view

| # | Feature | What it is |
|---|---|---|
| 1 | Elapsed days | Days since this card's previous review (the interval length) |
| 2 | Elapsed days, cumulative | Running sum of the card's intervals (card "calendar age") |
| 3 | Elapsed seconds | Seconds since this card's previous review |
| 4 | Elapsed-seconds sub-day phase | Where the interval falls within a 24 h day (e.g. 1.0-day vs 1.5-day gap) |
| 5 | Elapsed seconds, cumulative | Running sum of the card's elapsed seconds |
| 6 | Cumulative-seconds sub-day phase | 24 h phase of the cumulative clock |
| 7 | Review duration | Answer time of this review |
| 8 | Grade | Again / Hard / Good / Easy |
| 9 | Missing-ID flags | Note / deck / preset ID was missing |
| 10 | Days since any review | Days since the user's previous review of any card |
| 11 | Pseudo-day-of-week | Position in a 7-day cycle |
| 12 | New cards since card's last review | New cards the user reviewed for the first time since this card's previous review |
| 13 | Reviews since card's last review | Other reviews the user did since this card's previous review |
| 14 | New cards today | New cards done so far today |
| 15 | Reviews today | Reviews done so far today |
| 16 | Card state | Anki card state (new/learning/review/relearning) |
| 17 | Query flag | Marks the synthetic "predict cold" rows used by ahead mode |
| 18 | Card ID | ID of this exact card |
| 19 | Sibling (note) ID | ID of the note — siblings share it |
| 20 | Deck ID | ID of the deck |
| 21 | Preset ID | ID of the deck-options preset |
| 22 | 3-day cycle | Review day's position in a 3-day cycle |
| 23 | Pseudo-week cycle (7 d) | Same as above for a 7-day period |
| 24 | Pseudo-month cycle (30 d) | Same as above for a 30-day period |
| 25 | Pseudo-quarter cycle (100 d) | Same as above for a 100-day period |
| 26 | Pseudo-year cycle (365 d) | Same as above for a 365-day period |
| 27 | Pseudo-decade cycle (3650 d) | Same as above for a 3650-day period |
| 28 | Pseudo-century cycle (36500 d) | Same as above for a 36500-day period |

## Simplified view — 114-dim layout (`RWKV_ID_FEATURES=1`, the `-id` rebuild)

The `-id` databases (gen 3 onward; `featB` and every gen-4 run) feed a **114-dim** vector: the
92-dim layout above **minus Card state (#16, dropped at the source)** plus **23 real-timestamp
columns**, appended after the query flag. Source of truth: `rwkv/id_features.py` (`NEW_COLUMNS`,
`DROPPED_COLUMN`). Rows 1–15 and 36–46 are unchanged from the table above; the numbering shifts
by one because #16 is gone. Sin/cos pairs are one row, as above.

Two things to know when reading it. **Timezones are unknown** — `review_time` is epoch-ms with no
offset — so #17, #19 and #20 are UTC-based; #18 is the one that cancels the shift (deviation from
the user's own running mean) and is the high-value form. **Every count is clipped at review time**
(#29–#32): the model is never told how many cards the user went on to create later that day.

| # | Feature | What it is |
|---|---|---|
| 1 | Elapsed days | Days since this card's previous review (the interval length) |
| 2 | Elapsed days, cumulative | Running sum of the card's intervals (card "calendar age") |
| 3 | Elapsed seconds | Seconds since this card's previous review (**end-to-start**) |
| 4 | Elapsed-seconds sub-day phase | Where the interval falls within a 24 h day |
| 5 | Elapsed seconds, cumulative | Running sum of the card's elapsed seconds |
| 6 | Cumulative-seconds sub-day phase | 24 h phase of the cumulative clock |
| 7 | Review duration | Answer time of this review (zeroed for the most recent review at deploy) |
| 8 | Grade | Again / Hard / Good / Easy |
| 9 | Missing-ID flags | Note / deck / preset ID was missing |
| 10 | Days since any review | Days since the user's previous review of any card |
| 11 | Pseudo-day-of-week | Position in a 7-day cycle counted from the user's first day (see #19) |
| 12 | New cards since card's last review | New cards the user reviewed for the first time since this card's previous review |
| 13 | Reviews since card's last review | Other reviews the user did since this card's previous review |
| 14 | New cards today | New cards done so far today |
| 15 | Reviews today | Reviews done so far today |
| 16 | Query flag | Marks the synthetic "predict cold" rows used by ahead mode |
| 17 | Time of day | Where in the 24 h day the card was shown (UTC clock) |
| 18 | Time-of-day deviation | How far this review sits from the user's usual study hour (running circular mean of their earlier reviews; the unknown timezone cancels) |
| 19 | Day of week | True calendar weekday (UTC) — the real version of #11 |
| 20 | Day of year | True calendar day of year (UTC) — the real version of the 365-day cycle, #44 |
| 21 | Weekend flag | Saturday or Sunday |
| 22 | Seconds since any review | Seconds from the end of the user's previous review of any card to this card being shown — the sub-day version of #10 |
| 23 | User tenure | Time since the user's first-ever review |
| 24 | Creation → first review | How long the card existed before it was first reviewed |
| 25 | Deck age at review | How old the deck was when this review happened |
| 26 | Card-predates-deck flag | Card was created before the deck it now sits in (the normal case — cards move between decks) |
| 27 | Default-deck flag | Card is in Anki's default deck, where no deck age is defined |
| 28 | Deck depth | How deeply the deck is nested in the deck tree |
| 29 | Creation batch, 1 min | How many of the user's cards were created within a minute of this one (an import drops hundreds at once; a hand-made card is alone) |
| 30 | Creation batch, 1 h | Same for a one-hour window |
| 31 | Creation batch, 1 d | Same for a one-day window |
| 32 | Creation-batch position | How many cards were created in the hour *before* this one — its place within its batch |
| 33 | Default-preset flag | Deck uses Anki's default options preset (the user never configured it) |
| 34 | Sibling gap | Seconds since the end of the most recent review of a *different* card of the same note (past siblings only — a sibling's future reviews are never visible) |
| 35 | Card-predates-first-review flag | Card was created before the user ever reviewed anything (an imported or pre-existing collection) |
| 36 | Card ID | ID of this exact card |
| 37 | Sibling (note) ID | ID of the note — siblings share it |
| 38 | Deck ID | ID of the deck |
| 39 | Preset ID | ID of the deck-options preset |
| 40 | 3-day cycle | Review day's position in a 3-day cycle, **plus the same for the day this card was first reviewed** (a card-cohort anchor). The phase is an arbitrary fixed offset from the user's first day, so these encode *relative* position, never the calendar — see the note below the table |
| 41 | Pseudo-week cycle (7 d) | Same as above for a 7-day period |
| 42 | Pseudo-month cycle (30 d) | Same as above for a 30-day period |
| 43 | Pseudo-quarter cycle (100 d) | Same as above for a 100-day period |
| 44 | Pseudo-year cycle (365 d) | Same as above for a 365-day period |
| 45 | Pseudo-decade cycle (3650 d) | Same as above for a 3650-day period |
| 46 | Pseudo-century cycle (36500 d) | Same as above for a 36500-day period |

**Why rows 40–46 are still here when #19/#20 are real (Andrew 2026-09-02).** Partly scope: the
`-id` rebuild changed only the card-feature block (`CARD_FEATURE_COLUMNS`); the encoding block —
IDs plus these cycles, built in `prepare_batch.add_encodings` from `DAY_OFFSET_ENCODE_PERIODS`
in `rwkv/config.py` — was never revisited. That was an oversight, not a decision. But they are
also not duplicates of the real calendar: each period's phase is an arbitrary fixed offset from
the user's first day (a seeded `randint`), so a "7-day cycle" cannot say *which* weekday — only
*same phase as N days ago* — and each period also carries the card's **first-review day**, making
the pair a multi-scale card-cohort/age encoding. Only the 7 d and 365 d periods have real
counterparts at all; 3/30/100/3650/36500 do not, and the long ones overlap #23 (tenure) instead.
Whether the model still leans on them with real dow/doy and tenure available is empirical, and an
ablation arm (`abl_cycles`, checkpoint surgery on featB) is queued behind gen4base to answer it.
If they are dead weight, the next rebuild drops 28 dims (114 → 86).

**Superseded the same day by Andrew's directive: replace every pseudo cycle with its real
counterpart — and row 11 with it.** `RWKV_REAL_CYCLES=1` (default off; needs `RWKV_ID_FEATURES=1`
and a rebuild — generation 5) removes rows 40–46 from the encoding block **and drops row 11
(pseudo-day-of-week, whose real counterpart is #19)**, and adds **24 card-feature columns** after
row 35: for each period in {3, 7, 30, 100, 365.25, 3650, 36500} days, sin/cos of the
**epoch-anchored UTC day index** of the review (`cyc{N}_sin/cos`) and of the card's first review
(`cyc{N}_first_sin/cos`), with no random baseline — so the phase means the same thing for every
user. The review-time 7 d and 365 d halves are *not* duplicated (they are #19/#20). Input becomes
**109** (69 card features + 40 ID dims). Same math as the pseudo cycles, real clock instead of the
user-relative one; and, being card-feature columns, they are name-ablatable. The `realcyc` run
measures it against `gen4base`, size-gated and single-variable.

What `featB` measured about these (2026-09-02, vs the 92-dim control `featA2`): **+0.000303 ahead /
+0.002371 imm**, ~+0.00053 / +0.00273 once the end-to-start penalty inside the bundle is added
back. The gain was **not** concentrated in same-day users, which points at the always-defined rows
(#23–#35) rather than the clock rows (#17–#22); an ablation on featB's own checkpoint is queued to
settle which. Detail: `optimization/research_5k_verbose.md`, featB section.

## Future input features (for when the no-new-inputs invariant is lifted)

Moved to **[`optimization/FUTURE_FEATURES.md`](optimization/FUTURE_FEATURES.md)** — the
consolidated, prioritized list of features derivable from real Anki timestamps (card/note/deck
IDs and review IDs are epoch-ms creation/review times), cross-checked against this table so
nothing already covered gets re-added. Not possible on the anonymized benchmark dataset (no
absolute timestamps); needs a new dataset export.

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
- **Row 11 vs row 23 (both 7-day, NOT a duplicate):** row 11 is a single sawtooth
  (`data_processing.py::add_segment_features`) with a **fixed** phase (day_offset is
  re-zeroed to the segment's first day), current review day only — a stable weekly
  signal the net can rely on directly. Row 23 is the 7-day member of the sin/cos
  cycle family: smooth (no wrap discontinuity), **randomly re-phased every batch**,
  and it also encodes the card's first-review day. Same period, different phase
  stability + extra cohort info.
- **Labels** (training targets, not inputs; from the card's *next* review):
  `label_y`, `label_rating`, `label_elapsed_days`, `label_elapsed_seconds` — the
  forgetting-curve head is supervised at the actual next-interval point;
  `label_is_equalize` marks reviews that count in the benchmark.
- ⚠ **Invariant** (optimization protocol): the model must keep running on this exact
  92-dim preprocessed input / the existing LMDBs — no new or changed inputs.
