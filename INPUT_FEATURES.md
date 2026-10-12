# RWKV input features — the full 92-dim per-review vector

Source of truth: `rwkv/data_processing.py` (dense features + query masking),
`rwkv/prepare_batch.py` (ID + day-offset encodings, `add_encodings`),
`rwkv/config.py` (dims + periods). One conceptual feature per row; sin/cos pairs and
multi-dim encodings are counted as one feature (the `dims` column sums to **92**).
"log-z" = `(log(...) − mean)/std` with the constants in `STATISTICS`
(`data_processing.py`).

> **STATUS MARKS (Andrew 2026-09-12).** The final model uses the **final layout: 103 inputs** (105 since 2026-09-25, with the sibling rating)
> (63 card features + 40 ID dims; `RWKV_ID_FEATURES=1` + `RWKV_REAL_CYCLES=1` +
> `RWKV_FINAL_LAYOUT=1`). Andrew: *"both ❌ and ⚠️ should be removed for the Really Really For Real
> This Time Finalized Final, Finally! version of RWKV"*. In the tables below:
> - ❌ = **not in the final model** (removed at the source, replaced by its real-clock version, or
>   removed as a duplicate by `RWKV_FINAL_LAYOUT=1`, which slices 6 columns out of each gen-5 row as
>   it is read -- no rebuild).
> - ⚠️ = was "dead or a duplicate" in the 2026-09-09 feature audit. **Every ⚠️ item was re-measured on
>   the full data before removal (`scratchpad/final_layout/`), and one claim did not hold:** the
>   default-deck flag is NOT dead, so it STAYS (no mark now). The others are now ❌.
> - no mark = in the final model and useful.
>
> The full final list is the table **"Final layout — 105 inputs"** at the end of the 114-dim section
> (rows 44–45 of that table, the sibling rating, were accepted on 2026-09-25; the nine read-time columns added on
> 2026-09-22/23 were dropped on 2026-09-24 and are no longer in the tables).
>
> **UPDATE 2026-09-22/23: the final layout is now 112 inputs = the 103 above + 9 "read-time"
> columns: the eight rowfeat columns (Andrew: "rowfeat - accept", iteration 102) and cards reviewed at
> least once (Andrew: "Let's accept cum_new_cards", iteration 96).** They need no
> LMDB rebuild: `rwkv/extra_features.py` computes them as a row is read, from a per-row side table,
> behind `RWKV_EXTRA_FEATURES`. ❌ **DROPPED 2026-09-24 (Andrew: "Drop nine columns")**: the 10+1
> confirmation `s64hrf10` (iteration 109) gave imm +0.000150 but ahead -0.000209, significantly WORSE (p 1.4e-3),
> so it fails the either-mode gate. The shipped layout stayed at 103 inputs (`s64hc10`).
>
> **UPDATE 2026-09-25: the sibling rating (2 inputs, `sib_grade`, `sib_none`) is ACCEPTED (iteration 110 `s64hsm10`),
> so the incumbent layout is 105 inputs.** See "Andrew's other two kept candidates" below.

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
| 9 | Missing-ID flags (❌ deck + preset flags; the note flag stays) | 3 | Note / deck / preset ID was missing (missing notes get a unique per-card placeholder; missing decks/presets share one placeholder) | 0/1 each |
| 10 | Days since any review | 1 | Days since the user's previous review of **any** card | `log(log(e+x))` |
| 11 | ❌ Pseudo-day-of-week | 1 | Position in a 7-day cycle (phase arbitrary — day 0 is anonymized) | `((day_offset mod 7) − 3)/3` ∈ [−1,+1] |
| 12 | New cards since card's last review | 1 | How many **new** cards the user introduced between this card's previous review and now | `log(3+x)` log-z |
| 13 | Reviews since card's last review | 1 | How many **other** reviews the user did in that same window | `log(3+x)` log-z |
| 14 | New cards today | 1 | Running count of new cards introduced so far today | `log(3+x)` log-z |
| 15 | Reviews today | 1 | Running count of reviews done so far today | `log(3+x)` log-z |
| 16 | ❌ Card state | 1 | Anki card state (new/learning/review/relearning) | `state − 2` |
| 17 | Query flag | 1 | 1 on the synthetic "predict cold" rows used by ahead mode (see masking note below) | 0/1 |
| 18 | Card ID | 12 | Identity of this exact card | random code per entity, each dim uniform over {−1.5,−0.5,+0.5,+1.5}; **re-randomized every batch** (see note) |
| 19 | Note ID | 12 | Identity of the note — cards generated from the same note share it | 〃 |
| 20 | Deck ID | 8 | Identity of the deck | 〃 |
| 21 | Preset ID | 8 | Identity of the deck-options preset | 〃 |
| 22 | ❌ 3-day cycle | 4 | Position of the review day in a 3-day cycle, plus the same for the day this card was **first** reviewed (card-cohort anchor) | `sin`, `cos` × {review day, first-review day}; random per-batch phase baseline |
| 23 | ❌ Pseudo-week cycle (7 d) | 4 | 〃 for a 7-day period | 〃 |
| 24 | ❌ Pseudo-month cycle (30 d) | 4 | 〃 for a 30-day period | 〃 |
| 25 | ❌ Pseudo-quarter cycle (100 d) | 4 | 〃 for a 100-day period | 〃 |
| 26 | ❌ Pseudo-year cycle (365 d) | 4 | 〃 for a 365-day period | 〃 |
| 27 | ❌ Pseudo-decade cycle (3650 d) | 4 | 〃 for a 3650-day period | 〃 |
| 28 | ❌ Pseudo-century cycle (36500 d) | 4 | 〃 for a 36500-day period | 〃 |
| | **Total** | **92** | | |

## Simplified view

| # | Feature | Type | What it is |
|---|---|---|---|
| 1 | Elapsed days | int (days) | Days since this card's previous review (the interval length) |
| 2 | Elapsed days, cumulative | int (days) | Running sum of the card's intervals (card "calendar age") |
| 3 | Elapsed seconds | int (s) | Seconds since this card's previous review |
| 4 | Elapsed-seconds sub-day phase | float ×2 (sin, cos) | Where the interval falls within a 24 h day (e.g. 1.0-day vs 1.5-day gap) |
| 5 | Elapsed seconds, cumulative | int (s) | Running sum of the card's elapsed seconds |
| 6 | Cumulative-seconds sub-day phase | float ×2 (sin, cos) | 24 h phase of the cumulative clock |
| 7 | Review duration | int (ms) | Answer time of this review |
| 8 | Grade | bool ×4 (one-hot) | Again / Hard / Good / Easy |
| 9 | Missing-ID flags (❌ deck + preset flags; the note flag stays) | bool ×3 | Note / deck / preset ID was missing |
| 10 | Days since any review | int (days) | Days since the user's previous review of any card |
| 11 | ❌ Pseudo-day-of-week | int (0–6) | Position in a 7-day cycle |
| 12 | New cards since card's last review | int | New cards the user reviewed for the first time since this card's previous review |
| 13 | Reviews since card's last review | int | Other reviews the user did since this card's previous review |
| 14 | New cards today | int | New cards done so far today |
| 15 | Reviews today | int | Reviews done so far today |
| 16 | ❌ Card state | int (0–3) | Anki card state (new/learning/review/relearning) |
| 17 | Query flag | bool | Marks the synthetic "predict cold" rows used by ahead mode |
| 18 | Card ID | code, float ×12 | ID of this exact card |
| 19 | Note ID | code, float ×12 | ID of the note — siblings share it |
| 20 | Deck ID | code, float ×8 | ID of the deck |
| 21 | Preset ID | code, float ×8 | ID of the deck-options preset |
| 22 | ❌ 3-day cycle | float ×4 (sin, cos × review day, first-review day) | Review day's position in a 3-day cycle |
| 23 | ❌ Pseudo-week cycle (7 d) | float ×4 (sin, cos × review day, first-review day) | Same as above for a 7-day period |
| 24 | ❌ Pseudo-month cycle (30 d) | float ×4 (sin, cos × review day, first-review day) | Same as above for a 30-day period |
| 25 | ❌ Pseudo-quarter cycle (100 d) | float ×4 (sin, cos × review day, first-review day) | Same as above for a 100-day period |
| 26 | ❌ Pseudo-year cycle (365 d) | float ×4 (sin, cos × review day, first-review day) | Same as above for a 365-day period |
| 27 | ❌ Pseudo-decade cycle (3650 d) | float ×4 (sin, cos × review day, first-review day) | Same as above for a 3650-day period |
| 28 | ❌ Pseudo-century cycle (36500 d) | float ×4 (sin, cos × review day, first-review day) | Same as above for a 36500-day period |

## Simplified view — the `-id` layout (`RWKV_ID_FEATURES=1`; gen 5 with real cycles = 109 inputs, gen 2–4 = 114 with the pseudo cycles in rows 40–46)

The `-id` databases (gen 3 onward; `featB` and every gen-4 run) feed a **114-dim** vector: the
92-dim layout above **minus Card state (#16, dropped at the source)** plus **23 real-timestamp
columns**, appended after the query flag. Source of truth: `rwkv/id_features.py` (`NEW_COLUMNS`,
`DROPPED_COLUMN`). Rows 1–15 and 36–46 are unchanged from the table above; the numbering shifts
by one because #16 is gone. Sin/cos pairs are one row, as above.

Two things to know when reading it. **Timezones are unknown** — `review_time` is epoch-ms with no
offset — so #17, #19 and #20 are UTC-based; #18 is the one that cancels the shift (deviation from
the user's own running mean) and is the high-value form. **Every count is clipped at review time**
(#29–#32): the model is never told how many cards the user went on to create later that day.

| # | Feature | Type | What it is |
|---|---|---|---|
| 1 | Elapsed days | int (days) | Days since this card's previous review (the interval length) |
| 2 | Elapsed days, cumulative | int (days) | Running sum of the card's intervals (card "calendar age") |
| 3 | Elapsed seconds | int (s) | Seconds since this card's previous review (**end-to-start**) |
| 4 | Elapsed-seconds sub-day phase | float ×2 (sin, cos) | Where the interval falls within a 24 h day |
| 5 | Elapsed seconds, cumulative | int (s) | Running sum of the card's elapsed seconds |
| 6 | Cumulative-seconds sub-day phase | float ×2 (sin, cos) | 24 h phase of the cumulative clock |
| 7 | Review duration | int (ms) | Answer time of this review (zeroed for the most recent review at deploy) |
| 8 | Grade | bool ×4 (one-hot) | Again / Hard / Good / Easy |
| 9 | Missing-ID flags (❌ deck + preset flags; the note flag stays) | bool ×3 (final layout: ×1) | Note / deck / preset ID was missing |
| 10 | Days since any review | int (days) | Days since the user's previous review of any card |
| 11 | ❌ Pseudo-day-of-week | int (0–6) | Position in a 7-day cycle counted from the user's first day (see #19) |
| 12 | New cards since card's last review | int | New cards the user reviewed for the first time since this card's previous review |
| 13 | Reviews since card's last review | int | Other reviews the user did since this card's previous review |
| 14 | New cards today | int | New cards done so far today |
| 15 | Reviews today | int | Reviews done so far today (what the Review Heatmap add-on shows) |
| 16 | Query flag | bool | Marks the synthetic "predict cold" rows used by ahead mode |
| 17 | Time of day | float ×2 (sin, cos) | Where in the 24 h day the card was shown (UTC clock) |
| 18 | Time-of-day deviation | float ×2 (sin, cos) | How far this review sits from the user's usual study hour (running circular mean of their earlier reviews; the unknown timezone cancels) |
| 19 | Day of week | int (0–6) | True calendar weekday (UTC), Monday = 0 — the real version of #11. Fed to the model as a sin/cos pair (2 dims) so Sunday sits next to Monday |
| 20 | Day of year | int (1–366) | True calendar day of year (UTC), fed as a sin/cos pair (2 dims) — the real version of the 365-day cycle, #44. ⚠ The phase divides by 365.25, not by the year's real length, so the wrap is off by up to one day in a leap year; the exact form is queued for the next rebuild (`FUTURE_FEATURES.md` top) |
| 21 | Weekend flag | bool | Saturday or Sunday |
| 22 | Seconds since any review | float (s) | Seconds from the end of the user's previous review of any card to this card being shown — the sub-day version of #10 |
| 23 | User tenure | float (s) | Time since the user's first-ever review |
| 24 | Creation → first review interval | float (s) | How long the card existed before it was first reviewed |
| 25 | Deck age at review | float (s) | How old the deck was when this review happened |
| 26 | Card-predates-deck flag | bool | Card was created before the deck it now sits in |
| 27 | Default-deck flag | bool | Card is in Anki's default deck, where no deck age is defined |
| 28 | Deck depth | int | How deeply the deck is nested in the deck tree; 0 = top level (for example, "Deck1" has depth 0 and "Deck1::Deck2" has depth 1) |
| 29 | Creation batch, 1 min | int | How many of the user's cards were created within a minute of this one (an import drops hundreds at once; a hand-made card is alone) |
| 30 | Creation batch, 1 h | int | Same for a one-hour window |
| 31 | Creation batch, 1 d | int | Same for a one-day window |
| 32 | Creation-batch position | int | How many cards were created in the hour *before* this one — its place within its batch |
| 33 | Default-preset flag | bool | Deck uses Anki's default options preset (the user never configured it) |
| 34 | Sibling gap | float (s) | Seconds since the end of the most recent review of a *different* card of the same note (past siblings only — a sibling's future reviews are never visible) |
| 35 | Card-predates-first-review flag | bool | Card was created before the user ever reviewed anything (an imported or pre-existing collection) |
| 36 | Card ID | code, float ×12 | ID of this exact card |
| 37 | Note ID | code, float ×12 | ID of the note — siblings share it |
| 38 | Deck ID | code, float ×8 | ID of the deck |
| 39 | Preset ID | code, float ×8 | ID of the deck-options preset |
| 40 | 3-day cycle (real) | float ×4 (sin, cos × today, first-review day) | Two clocks with a 3-day period. **Today's clock:** where today falls in the 3-day cycle. **The card's clock:** where the day of this card's FIRST review fell in the same cycle — a fixed property of the card, the same on every one of its rows (it tells the model "this card entered on a day with phase X", a cohort anchor). Real = the cycle counts from the epoch UTC day, so the phase means the same for every user. ❌ Replaced the pseudo 3-day cycle, whose phase was an arbitrary offset from the user's first day |
| 41 | Week cycle (real), card's clock only | float ×2 (sin, cos; first-review day) | The weekday on which this card was FIRST reviewed — a fixed property of the card. Today's weekday is #19, so it is not repeated here. Example: a card first seen on a Sunday carries "Sunday" in #41 on every later review, while #19 says what today is. ❌ Replaced the pseudo-week cycle |
| 42 | Month cycle (real, 30 d) | float ×4 (sin, cos × today, first-review day) | Same two clocks as #40 with a fixed 30-day period counted from the epoch — it drifts against the calendar month by about half a day per month, so it is not the calendar day of month. ⚠ To be REPLACED at the next rebuild by the calendar-exact day of month (28/29/30/31-day months; `FUTURE_FEATURES.md` top). ❌ Replaced the pseudo-month cycle |
| 43 | Quarter cycle (real, 100 d) | float ×4 (sin, cos × today, first-review day) | Same two clocks as #40 with a fixed 100-day period — not the calendar quarter. ⚠ To be REPLACED at the next rebuild by the position inside the calendar quarter (90/91/92 days). ❌ Replaced the pseudo-quarter cycle |
| 44 | Year cycle (real, 365.25 d), card's clock only | float ×2 (sin, cos; first-review day) | The day of the year on which this card was FIRST reviewed — a fixed property of the card. Today's day of year is #20, so it is not repeated here. ⚠ Phase divides by 365.25; to be made leap-year exact at the next rebuild. ❌ Replaced the pseudo-year cycle |
| 45 | Decade cycle (real, 3650 d), today's clock only | float ×2 (sin, cos; today) | Where today falls in a fixed 3650-day period. With ~10 years of data this is close to a monotone "which year is it". The card's clock (first-review day) is ❌ in the final layout: over one user's history it sits at almost the same phase as today's (r = 0.99). ⚠ To be REPLACED at the next rebuild by the position inside the calendar decade. ❌ Replaced the pseudo-decade cycle |
| 46 | Century cycle (real, 36500 d), today's clock only | float ×2 (sin, cos; today) | Where today falls in a fixed 36500-day period; its cos is almost constant over the data (std 0.011). The card's clock is ❌ in the final layout (r = 0.99). ⚠ To be REPLACED at the next rebuild by the position inside the calendar century. ❌ Replaced the pseudo-century cycle |
| 47 | Sibling grade | int (1–4) | The grade of the latest review of ANOTHER card of the same note earlier in the same sitting (a just-failed sibling predicts failure); 0 = none. ACCEPTED 2026-09-25 (iteration 110 `s64hsm10`, with Muon-on-scale: imm +0.000105, ahead null vs `s64hc10`) |
| 48 | No-sibling flag | bool | 1 when no other card of this note was reviewed earlier in the same sitting (~96% of reviews) |

**Rows 47–48 (the sibling rating, ACCEPTED 2026-09-25, iteration 110)** need no LMDB rebuild: `rwkv/extra_features.py`
computes them as a row is read (`RWKV_EXTRA_FEATURES=sib_grade,sib_none`), using only reviews BEFORE the one being
predicted. The nine read-time columns that were here as rows 47–55 (dropped 2026-09-24, iteration 109) are removed from
the tables; they are described under "The eight read-time columns" below. Final-layout numbering: the
**"Final layout — 105 inputs"** table below (rows 44–45 there).

**Type column (added 2026-09-13).** The type is the quantity's own type before scaling; every input reaches the model as a float. `int` counts and durations are log-scaled and standardized; `bool` flags are 0/1; a `sin, cos` pair encodes a phase; `code` = a random per-entity code, re-drawn every batch (not the ID's value). Rows 40–46 are one progression — 3 days / week / month / quarter / year / decade / century. **Each period has two clocks: TODAY's clock (where the review day falls in the cycle) and the CARD's clock (where the day of the card's first review fell in the same cycle, a fixed property of the card, the same on all its rows).** For the week and the year, today's clock is already #19 / #20, so rows 41 and 44 hold only the card's clock; for the decade and the century the card's clock was dropped as a duplicate. Each real (epoch-anchored UTC day).


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

## Final layout — 105 inputs (what the incumbent model reads)

> Rows 1–43 (103 inputs) are the layout every run since `w10lf` trained on (`s64hc10`). Rows 44–45, the sibling rating, were
> ACCEPTED on 2026-09-25 (iteration 110 `s64hsm10`), so the incumbent reads 105 inputs. The nine read-time columns
> accepted on 2026-09-22/23 and DROPPED on 2026-09-24 (iteration 109: ahead significantly worse at 10+1) are not listed;
> see **"The eight read-time columns"** under the table.

Checked against the code on 2026-09-12 (`CARD_FEATURE_COLUMNS` under `RWKV_ID_FEATURES=1` +
`RWKV_REAL_CYCLES=1` + `RWKV_FINAL_LAYOUT=1` = 63 columns; the ID block is 40 dims). It is the
gen-5 layout (the 114-dim table above minus #11 and #40–46, plus 24 **real-calendar** cycle columns,
whose phase comes from the epoch-anchored UTC day index, so it means the same thing for every user)
minus the 6 duplicate columns listed under the table. The gen-5 LMDBs still store all 69 card
columns; `rwkv/prepare_batch.get_data` drops the 6 as it reads a row (`rwkv/id_features.py`
`FINAL_LAYOUT_DROPPED`; smoke `scratchpad/final_layout/smoke_final_layout.py`).

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
| 38 | Decade cycle (real, 3650 d): today's clock only | 2 | float ×2 (sin, cos) | the first-review half is ❌ (r = 0.99 with this half); a fixed 3650-day beat, calendar-exact version queued |
| 39 | Century cycle (real, 36500 d): today's clock only | 2 | float ×2 (sin, cos) | the first-review half is ❌ (r = 0.99); this cos is almost constant (std 0.011); a fixed 36500-day beat, calendar-exact version queued |
| 40 | Card ID code | 12 | code, float ×12 |  |
| 41 | Note ID code | 12 | code, float ×12 |  |
| 42 | Deck ID code | 8 | code, float ×8 |  |
| 43 | Preset ID code | 8 | code, float ×8 |  |
| 44 | Sibling grade | 1 | int (1–4) | grade of the latest review of another card of the same note earlier in the same sitting (30-min sitting cut); 0 = none; z-scored over the 4.28% of train rows that have one. `sib_grade` in `RWKV_EXTRA_FEATURES`, from `sib_flat.npy`. ACCEPTED 2026-09-25 (iteration 110) |
| 45 | No-sibling flag | 1 | bool | 1 when no other card of this note was reviewed earlier in the same sitting. `sib_none`. ACCEPTED 2026-09-25 (iteration 110) |
| | **Total (incumbent)** | **105** | | 65 card features + 40 ID dims (rows 1–45). Without the sibling rows: 103 (`s64hc10`) |

**The calendar cycles, per period** (Andrew 2026-09-25: "some are today, some are first-review, some are both"). Every
cycle has a TODAY half; only the decade and century lack a FIRST-REVIEW half:

| cycle | today half | first-review half |
|---|---|---|
| 3 d | row 33 | row 33 |
| 7 d (week) | row 18 (day of week) | row 34 |
| 30 d | row 35 | row 35 |
| 100 d | row 36 | row 36 |
| 365.25 d (year) | row 19 (day of year) | row 37 |
| 3650 d (decade) | row 38 | removed (r = 0.99 with the today half) |
| 36500 d (century) | row 39 | removed (r = 0.99) |

⚠ Two conventions are mixed (`rwkv/id_features.py:598-603` vs `:715-732`): rows 18-19 use the CALENDAR (Monday = 0;
day of year from 1, divided by 365.25), the first-review halves use the UTC day index counted from 1970-01-01 (a
Thursday) mod the period. So the week halves differ by a fixed 3-day phase and the year halves by 0-1 day (leap
years). No information is lost (a fixed phase offset of a sin/cos pair is a rotation the input layer absorbs), but the
Anki builder must copy both conventions exactly. Unify at the next rebuild (one definition for every half).

**❌ Removed from the gen-5 layout by `RWKV_FINAL_LAYOUT=1` (6 columns), each claim re-measured first:**

| column | why it is a duplicate | measured |
|---|---|---|
| `deck_id_is_nan` | equals `note_id_is_nan` **by construction**: note_id and deck_id both come from the one card-row join (int64 columns), so both are NaN exactly when the card row is missing | 0 differences in 16.9 M stored rows |
| `preset_id_is_nan` | equals `note_id_is_nan` except when a card exists but its deck has no row in the decks table | differs on **11,095 of 727.7 M reviews (0.0015%)**, 77 users (`preset_join_scan.py`, all 10,000 users) -- a near copy |
| `cyc3650_first_sin/cos` | the first-review half of the decade cycle; over one user's history the review day and the card's first day sit at almost the same phase | r = 0.9935 with the review half (audit W5) |
| `cyc36500_first_sin/cos` | the same for the century cycle | r = 0.9910 (audit W5) |

Params: 563,652 -> **561,732** (6 x 320 input-layer weights). The first run on this layout is `w10lf`,
the base of the shipped model.

### The eight read-time columns (accepted 2026-09-22, ❌ dropped 2026-09-24, not in the tables)

**Source:** `rwkv/extra_features.py` (`RWKV_EXTRA_FEATURES=sess_pos,sess_secs,roll10,roll60,phase_r,phase_r2,streak,grade_mean`),
from two per-row tables built once from `anki-revlogs-10k-id` by `scratchpad/rowtables/build_rowtables.py`:
`secs_flat.npy` (the SHOW time of every review, epoch seconds) and `rating_flat.npy` (its grade).
Normalisation constants are the train half's (`ROWFEAT_STATS`, 357,460,182 rows).

**What every column obeys:**
- **Only earlier reviews.** Each value uses the user's reviews strictly before the one being predicted,
  so the query row (the imm prediction) carries the same value as its real row. The leak check for the
  mean grade (the one column built from labels) is a within-user permutation floor, and it passes
  (`scratchpad/rowtables/`).
- **Computed from the user's first review, always** -- never from the start of a training chunk.
  Training splits a user into a median of 4 chunks and eval into 1, so a counter started at a chunk
  boundary would be a different quantity in training than at eval and in Anki.
- **The first review** has no history: rows 48, 49 and 51 then take the training mean (scaled 0).

**Measured value (102 `rowfeat`, s80 trunk, 1+1 budget, n = 2,485):** **+0.000087 ahead** (p 0.004) /
**+0.000171 imm** (p 3e-32). The gate needs both modes; Andrew accepted it on the imm gain. Effects at
the 1+1 budget run ~0.65 of their full-budget size.

**Params:** +320 per column on the 561k trunk (561,732 -> 564,292); **+128 per column on the shipped
s64h trunk (332,922 -> 333,946, measured)**.

**Status of the three paths** (AGENTS.md §9, train vs eval vs deploy must compute one quantity):

| path | computes the columns? |
|---|---|
| training + eval (`prepare_batch.get_data`) | yes |
| Python deploy RNN (`run_as_rnn`) | yes -- `deploy_vs_lmdb.py` shows 0 of 8 columns disagreeing (users 5030 / 5033) |
| Rust engine (`rust/rwkv-infer`) | **no -- owed** |
| Anki fork (Clanki) | **no -- owed**; it needs the user's review times and grades in order, which the scheduler already reads |

**Next:** `s64hrf10` (the shipped s64h trunk + these columns, 10 WS + 1 decay epochs, control `s64hc10`)
ran on the rented box with cards-reviewed-at-least-once included (all nine columns, 334,074 params). **Result (iteration 109): ahead -0.000209 (worse, p_worse 1.4e-3) / imm +0.000150 (p 3.9e-18) vs `s64hc10`, so it fails the either-mode gate. Andrew 2026-09-24: "Drop nine columns". The deploy port is not owed.**

### Andrew's other two kept candidates -- NOT in the final layout (status 2026-09-25)

| candidate | state | measured |
|---|---|---|
| **Total cards reviewed at least once** (`cum_new_cards`) | **ACCEPTED 2026-09-23** (Andrew: "Let's accept cum_new_cards even if it failed the p-value test"), then ❌ **DROPPED 2026-09-24** with the eight rowfeat columns: the 10+1 confirmation `s64hrf10` (iteration 109) made ahead significantly worse | 96 `newc1`: +0.000112 ahead / +0.000104 imm, both over the 0.0001 bar; ahead's p 3.2e-03 missed 1e-4. 98 `newc2` (with two more counters) was null. |
| **Sibling rating** (kept 2026-09-20, `optimization/FEATURE_IDEAS_2026-09-20.md` item D: the grade of a sibling card -- another card of the same note -- reviewed EARLIER IN THE SAME SITTING, plus a "no sibling this sitting" flag; 2 inputs. The idea: a just-failed sibling predicts this card's failure) | **BUILT 2026-09-23 and queued as the arm `s64hsib`** (`RWKV_EXTRA_FEATURES=sib_grade,sib_none`; `scratchpad/siblingrating/PREREG.md`). Precomputed table `sib_flat.npy` (int8, 538.6 MB, byte-identical on this PC and the box); deploy computes it online; `deploy_vs_lmdb.py` 0 disagreeing columns on users 5100/5076/5033, and a table shifted by one row makes both columns disagree on 20-31% of rows. Appended LAST in the column order. **Results:** 106 `s64hsib` at 1+1 on s64h: imm +0.000098 (p 1.3e-9), ahead null (-0.000003) -- rejected under the old both-modes rule by 0.000002, but it passes the relaxed either-mode gate of 2026-09-24. **ACCEPTED 2026-09-25 at 10+1 in 110 `s64hsm10`** (s64hc10 + these 2 columns + Muon on the scale matrices, 333,178 params): imm +0.000105 (p 7.4e-11), ahead -0.000011 (not significantly worse) vs s64hc10. **The incumbent layout is now 105 inputs = rows 1–43 + these 2.** Owed before it ships: the deploy port (Rust engine + the Anki fork; the Python RNN path already computes them). | 200-user screen: exists on **3.66%** of scored reviews; a failed sibling means a **0.37** fail rate vs 0.18; gain over a simple baseline **+0.000778**, positive on the sibling rows in **9 of 10** held-out-user splits. Train half: 4.28% of rows have one. |

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
