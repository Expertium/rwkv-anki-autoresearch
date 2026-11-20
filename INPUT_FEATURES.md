# RWKV input features - current 105-dim per-review vector

This inventory lists only inputs used by the current incumbent: 63 base card-feature dimensions, 40 ID dimensions, and 2 sibling-rating dimensions.

Source of truth: `rwkv/id_features.py`, `rwkv/extra_features.py`, `rwkv/prepare_batch.py`, and `rwkv/config.py`.

## Current layout - 105 inputs

The model uses `RWKV_ID_FEATURES=1`, `RWKV_REAL_CYCLES=1`, `RWKV_FINAL_LAYOUT=1`, and `RWKV_EXTRA_FEATURES=sib_grade,sib_none`. The gen-5 LMDB stores 69 card columns; `prepare_batch.get_data` removes six duplicate columns as it reads each row.
| # | Feature | Dims | Type | Note |
|---|---|---|---|---|
| 1 | Elapsed days | 1 | int (days) |  |
| 2 | Elapsed days, cumulative | 1 | int (days) |  |
| 3 | Elapsed seconds (end-to-start) | 1 | int (s) |  |
| 4 | Elapsed-seconds sub-day phase | 2 | float ×2 (sin, cos) |  |
| 5 | Elapsed seconds, cumulative | 1 | int (s) |  |
| 6 | Cumulative-seconds sub-day phase | 2 | float ×2 (sin, cos) |  |
| 7 | Review duration | 1 | int (ms) | zeroed for the review being predicted (it does not exist yet at show time) |
| 8 | Grade | 4 | bool ×4 (one-hot) |  |
| 9 | Missing-ID flag (note) | 1 | bool | 1 when the review's card row is missing from the export (a deleted card: ~26% of all -id reviews) |
| 10 | Days since any review | 1 | int (days) |  |
| 11 | New cards since card's last review | 1 | int |  |
| 12 | Reviews since card's last review | 1 | int |  |
| 13 | New cards today | 1 | int |  |
| 14 | Reviews today | 1 | int |  |
| 15 | Query flag | 1 | bool |  |
| 16 | Time of day (UTC) | 2 | float ×2 (sin, cos) |  |
| 17 | Time-of-day deviation | 2 | float ×2 (sin, cos) |  |
| 18 | Day of week (UTC) | 2 | int (0–6) | fed as a sin/cos pair; also the review-day half of the real 7-day cycle |
| 19 | Day of year (UTC) | 2 | int (1–366) | fed as a sin/cos pair; also the review-day half of the real 365.25-day cycle; phase = doy/365.25, so up to one day off at the wrap in a leap year (exact form queued) |
| 20 | Weekend flag | 1 | bool |  |
| 21 | Seconds since any review | 1 | float (s) | for in-session gaps the query and probe rows see the previous answer's time (the clock fix, `RWKV_CLOCK_AT_PREV_ANSWER`) |
| 22 | User tenure | 1 | float (s) |  |
| 23 | Creation → first review interval | 1 | float (s) | reliance measured 2026-09-12 (w10clk, zeroed, n=300): +0.0002 ahead / +0.0002 imm on all rows; **+0.0007 / +0.0006 on each card's SECOND review** (2026-09-14, the first event the model predicts — a first review is never predicted), so it matters most for new cards. A ceiling on its value, since a retrained model would compensate |
| 24 | Deck age at review | 1 | float (s) |  |
| 25 | Card-predates-deck flag | 1 | bool |  |
| 26 | Default-deck flag | 1 | bool | set on 14 of 100 sampled users (up to 100% of one user's rows); on those rows the deck-missing flag is 0 and deck age is 0, so this flag is the only thing that tells Anki's default deck from a deck created at review time. The audit's "zero on every row" was a 17-user sample |
| 27 | Deck depth | 1 | int |  |
| 28 | Creation batch, 1 min / 1 h / 1 d | 3 | int ×3 |  |
| 29 | Creation-batch position | 1 | int |  |
| 30 | Default-preset flag | 1 | bool |  |
| 31 | Sibling gap | 1 | float (s) | undefined on ~97% of rows and coded as 0.0, which is also a legal value (audit W1). The flag that separates the two cases was TESTED 2026-09-26 as a read-time input (`sibgap_none`, iteration 113 `s64hsg10`, 10+1 vs `s64hsm10`): a clean null (+0.000037 ahead / -0.000010 imm, p 0.64 / 0.74) -> NOT adopted, and not needed at the next rebuild. The model already tells the cases apart (rows 44-45 and the note stream) |
| 32 | Card-predates-first-review flag | 1 | bool |  |
| 33 | 3-day cycle (real): today's clock + the card's clock | 4 | float ×4 (sin, cos × review day, first-review day) |  |
| 34 | Week cycle (real): the card's clock only (the weekday of its first review) | 2 | float ×2 (sin, cos) | today's clock = #18 (day of week) |
| 35 | Month cycle (real, 30 d): today's clock + the card's clock | 4 | float ×4 (sin, cos × review day, first-review day) | a fixed 30-day beat from the epoch, not the calendar month (drifts ~half a day per month); calendar day of month queued for the next rebuild |
| 36 | Quarter cycle (real, 100 d): today's clock + the card's clock | 4 | float ×4 (sin, cos × review day, first-review day) | a fixed 100-day beat, not the calendar quarter; calendar-exact version queued for the next rebuild |
| 37 | Year cycle (real, 365.25 d): the card's clock only (the day of year of its first review) | 2 | float ×2 (sin, cos) | today's clock = #19 (day of year); phase /365.25, leap-year-exact version queued |
| 38 | Decade cycle (real, 3650 d): today's clock only | 2 | float ×2 (sin, cos) | The first-review cycle is omitted because it correlates 0.99 with today's cycle. A fixed 3650-day period; calendar-exact version planned. |
| 39 | Century cycle (real, 36500 d): today's clock only | 2 | float ×2 (sin, cos) | The first-review cycle is omitted because it correlates 0.99 with today's cycle. The cosine varies little (std 0.011). A fixed 36500-day period; calendar-exact version planned. |
| 40 | Card ID code | 12 | code, float ×12 |  |
| 41 | Note ID code | 12 | code, float ×12 |  |
| 42 | Deck ID code | 8 | code, float ×8 |  |
| 43 | Preset ID code | 8 | code, float ×8 |  |
| 44 | Sibling grade | 1 | int (1–4) | grade of the latest review of another card of the same note earlier in the same sitting (30-min sitting cut); 0 = none; z-scored over the 4.28% of train rows that have one. `sib_grade` in `RWKV_EXTRA_FEATURES`, from `sib_flat.npy`. ACCEPTED 2026-09-25 (iteration 110) |
| 45 | No-sibling flag | 1 | bool | 1 when no other card of this note was reviewed earlier in the same sitting. `sib_none`. ACCEPTED 2026-09-25 (iteration 110) |
| | **Total (incumbent)** | **105** | | 65 card features + 40 ID dims (rows 1–45). Without the sibling rows: 103 (`s64hc10`) |

**Calendar cycle rows:**
The table lists the clock halves used by the current model.

| cycle | active rows |
|---|---|
| 3 d | row 33: today and first-review |
| 7 d (week) | row 18: today; row 34: first-review |
| 30 d | row 35: today and first-review |
| 100 d | row 36: today and first-review |
| 365.25 d (year) | row 19: today; row 37: first-review |
| 3650 d (decade) | row 38: today |
| 36500 d (century) | row 39: today |

⚠ Two conventions are mixed (`rwkv/id_features.py:598-603` vs `:715-732`): rows 18-19 use the CALENDAR (Monday = 0;
day of year from 1, divided by 365.25), the first-review cycle halves use the UTC day index counted from 1970-01-01 (a
Thursday) mod the period. So the week halves differ by a fixed 3-day phase and the year halves by 0-1 day (leap
years). No information is lost (a fixed phase offset of a sin/cos pair is a rotation the input layer absorbs), but the
Anki builder must copy both conventions exactly. The Anki builder must match these conventions.

## Notes
- **Query masking:** ahead-mode query rows zero the current review's grade and duration. Timing, IDs, and counters remain available. See `add_queries` in `rwkv/data_processing.py`.
- **ID codes:** card, note, deck, and preset codes are random per-entity values, redrawn each batch. Matching codes identify repeated entities. These codes also route rows into the recurrent streams.
- **Cycle phases:** calendar cycles use UTC day indices. The table above describes their active clock halves. The Anki builder must match their phase conventions.
- **Labels are targets, not inputs:** `label_y`, `label_rating`, `label_elapsed_days`, and `label_elapsed_seconds` come from the next review. `label_is_equalize` marks reviews counted by the benchmark.
- **Future features:** see [`optimization/FUTURE_FEATURES.md`](optimization/FUTURE_FEATURES.md).
