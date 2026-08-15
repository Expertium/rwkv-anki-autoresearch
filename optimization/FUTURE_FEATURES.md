# Input features from real timestamps — THE DATASET IS BUILT; this is now an implementation plan

> **★ STATUS CORRECTION 2026-07-26.** This page was written as "planning only — needs a new
> dataset export". **That export was built the very next day and this header was never updated**,
> which made the whole family look blocked (it briefly did to Claude again on 2026-07-26 until
> Andrew said *"there is `anki-revlogs-10k-id` ... we should have code for making it"*).
> **`C:\Users\Andrew\anki-revlogs-10k-id` exists**, built by `scratchpad/dataset_id/`
> (`run_build_id.cmd` -> `build_parquet_id.py`): 10,000 user dirs, raw Anki epoch-ms IDs, and
> `review_time` corrected to SHOW time (`revlog.id - taken_millis`). Spot-checked on user 1:
> `review_time` 2021-05-22 15:31:47 UTC, `card_id` 15:14:10 UTC.
> **=> every "high" row below is derivable TODAY.** What remains is a preprocessing pass + an LMDB
> rebuild sourced from `-id` (⚠ plan disk: `train_db_5k_h1` is 372.5 GB, C: has 229 GB free, F: has
> 890 GB). The DECK TREE section needs no rebuild at all — see its own correction.

**Context (Andrew + Claude, 2026-07-15).** anki-revlogs-10k is anonymized: `day_offset`
integer days only, re-indexed IDs. In real Anki, `card_id` / `note_id` / `deck_id` /
`review_id` are all **epoch-millisecond creation/review timestamps**, so a fresh export that
preserves them unlocks the features below — **and that export now exists (see above)**. This
breaks the current 92-dim input invariant → a **data-side phase**, not something to A/B on the
current LMDBs. Deploy-side cost is
~zero (Anki has the real IDs at inference). Cross-checked against the existing 92-dim feature
table — [`INPUT_FEATURES.md`](../INPUT_FEATURES.md) at the repo root — so we don't re-add what
exists; the `#N` references below are its row numbers.

## Already covered by existing features — do NOT re-add
- Workload today: #14 (new cards today), #15 (reviews today).
- Since this card's last review: #12/#13 (new cards / reviews in between).
- Cross-card recency at day granularity: #10 (days since any review).
- Sub-day phases of this card's own clock: #4/#6.
- Cyclic calendar structure at 3/7/30/100/365/3650/36500 d: #11 + #22–28 — but
  **pseudo-phase** (anchored to day_offset, arbitrary per user).

## Candidate new features (consolidated after Andrew's cross-check)
| Priority | Feature | Notes |
|---|---|---|
| high | Time-of-day: raw sin/cos of the 24 h phase, plus **user-relative deviation from a running *circular mean*** (per-user state = 2 floats: S += sin θ, C += cos θ over all prior reviews; usual hour = atan2(S,C)) | Andrew's #1; sidesteps the unknown-timezone problem (a timezone = constant phase offset, cancels in the deviation). Circular mean replaces the original "median hour" idea (Andrew's efficiency concern 2026-07-16): O(1)/review, 8 B/user, and circular-correct where a median breaks for around-midnight reviewers. Plain running mean, NO decay (Andrew 2026-07-16: EMA not needed). Fallback worth A/B-ing: raw phase only — the recurrent user stream can learn "usual hour" internally. |
| high | **Real-phase** calendar cycles (true day-of-week/month/year/decade, sin/cos) | Andrew's #2; upgrades #11/#22–28 from pseudo- to true phase → shared weekend/holiday effects across users. Weekend/weekday binary as the cheap special case (👍). |
| high | First review − card creation | Andrew's #3; completes card age: #2/#5 count from FIRST REVIEW, this covers creation→first-review. |
| high | Seconds-resolution "time since any review" (session position) | #10 is integer-day (built from day_offset) → sub-day session structure is invisible today. Continuous gap ≫ arbitrary session-split heuristics. |
| med | Creation-batch size at ±1 min / ±1 h / same day (+ position in batch) | Andrew's #4 generalized; import-vs-handmade signal. Andrew 👍 |
| med | User tenure (time since user's first-ever review) | Confirmed NOT in the table. |
| med | note_id/deck_id/preset_id ages: card − deck creation, card - preset creation, deck age at review, preset age at review | Early core card vs late addition; preset ids are creation timestamps too (Andrew 2026-07-16: use both). ⚠ the DEFAULT deck and DEFAULT preset both have id 1 (constant, not a timestamp) — derive an is-default flag for those instead of an age. Andrew 👍 **Coverage MEASURED 2026-07-27, and it splits the row in two — see below.** |

### Coverage of the id-age features, measured (300 users, 59,285 deck rows, `-id` set)
The `id == 1` default case the row above anticipated is **the entire non-timestamp population** —
the only small `preset_id` value that occurs at all is `1` (in 296 of 300 users). So these ids are
cleanly bimodal: real epoch-ms creation stamp, or the default sentinel. Nothing in between, no
parsing risk. But the two ids have very different coverage, and that changes what each dim is worth:

| id | timestamp-like | what that means for the feature |
|---|---|---|
| `deck_id` | **99.5%** of deck rows | deck age / card−deck-creation are **full-coverage** features. Build them. |
| `preset_id` | **7.0%** of deck rows (34.3% of users have ≥1) | "preset age" is defined for 1 row in 14. Ship the **is-default-preset flag** (93% vs 7% — a real split, and it says the user bothered to configure this deck); treat preset AGE as a low-value add-on, not a peer of deck age. |

Consistent with the degeneracy finding below (67.4% of users have exactly one preset): most users
never leave the default preset, so most of what a preset-age dim could carry is already carried by
the flag. `review_time` and `card_id`/`note_id` are timestamps for **100%** of rows (5 users
spot-checked end to end), so the high-priority rows are unaffected by any of this.
| low | Card created before vs after user's first-ever review | "Probably not important, but we can try" (Andrew). |
| skip | card_id − note_id gap | ~always zero (cards generated at note creation) — not worth a dim. |
| skip | Session count per day | Splitting is arbitrary; the sub-day #10 upgrade carries the signal continuously. |

## ⚠ DECK TREE — available TODAY, no new export needed (Andrew's find, 2026-07-24)

`DeckEntry.parent_id` (stats.proto field 2, present in BOTH the raw `-id` rebuild and the
PUBLISHED anonymized dataset) is **the parent deck's `deck_id`** — Anki's `A::B::C` deck
tree, one row per deck. Verified on the raw set (300-user sample, 58,552 deck rows):
94.8% of decks have a non-zero parent; **100% of those resolve to a `deck_id` in the SAME
user's table**; zero cycles, zero self-parents; `parent_id == 0` = top-level; depth up to
11 levels (mean per-user max 2.54). Parent is usually older than child (51,533 vs 3,967 —
the inversions are Anki auto-creating a parent when a deck is renamed into a new path),
and 98% of children share their parent's preset.

**The PUBLISHED set preserves it too** (200-user sample, 39,179 rows): `parent_id` was
factorized with the SAME codebook as `deck_id`, so 94.2% still resolve to real deck rows
and the depth profile matches; the `0` root sentinel became a per-user code that isn't a
deck (that's the 5.8% "unresolvable" = top-level decks). **So deck-hierarchy features need
NO new dataset export** — unlike everything else on this page. Our pipeline simply throws
it away: `rwkv/data_processing.py:203` does `df_decks.drop(columns=["user_id", "parent_id"])`
(inherited from upstream). ~~Cost to use it = an LMDB rebuild, not a data rebuild.~~

### ★ CORRECTION 2026-07-26 — it needs NO LMDB REBUILD EITHER. Verified end-to-end.

The "rebuild" estimate assumed the per-stream grouping is baked into the LMDB. It is not — the
grouping is DERIVED from a per-review id array, and that array is the **raw `deck_id`**:

1. `data_processing.py:459` stores `ids[submodule] = section_df[submodule]` — the dataset's own
   `deck_id` column cast to int32, not a remapped index. Key
   `{user}_{start}-{end}_{len}_deck_id_id_`.
2. **Empirically confirmed** (user 1, first chunk, 15,191 reviews): 14 unique stored deck ids,
   **14/14 resolve to real `deck_id` rows in that user's `decks` parquet**, and all 14 carry a
   non-zero `parent_id`.
3. `prepare_batch.insert_probes` ALREADY rebuilds a submodule's entire `ModuleData`
   (`split_len` / `split_B` / `from_perm` / `to_perm`) from an id array in ~20 lines of numpy, at
   batch time, on every probe-inserted sample. Ancestor levels are the same call on
   `parent_of(...)` applied to the stored ids.

**=> a per-user `deck_id -> parent_id` map (38 rows for user 1; a few MB across 10k users) plus
the existing grouping code gives ancestor-level groupings with the LMDBs untouched.** What this
removes is not small: the original build's ETA was **2-4 days of CPU**, and `train_db_5k_h1` is
**372.5 GB against 229 GB free on C:** — a side-by-side rebuild was never actually possible; it
would have meant deleting the only copy first, with no rollback.

Cost moves to the fetch workers (CPU per batch), which have headroom — the speed notes record
fetching as already fully hidden behind the GPU step, not a lever.

**Resolve rate CONFIRMED AT SCALE 2026-07-26** (`parent_id_probe3.py 2000`, now takes N on argv):
**2,000 users / 425,429 deck rows -> 94.5% of `parent_id`s resolve to a `deck_id` in the same
user's table**, matching the 200-user 94.2% at 10x the sample. **0 self-parents, 0 cycles.** The
5.5% miss is exactly the top-level decks, whose `0` root sentinel was factorized into a per-user
code that is not a deck — i.e. "no parent", not a data error. Deck-weighted depth histogram:
23,605 roots, then 45,760 / 81,449 / 116,058 / 85,627 / 42,359 at depths 1-5, tailing to 209 at
depth 11 (deck-weighted; the REVIEW-weighted profile above is the one that sets the loop cost).

So the deck tree is usable on the CURRENT LMDBs, unmodified. ⚠ Still spot-checked on one user for
the LMDB-id -> parquet link itself (14/14); widen that if anything looks off when the gathers are
built. Note the rebuild path would resolve at ~100% instead, since `-id` keeps real deck ids —
so if the timestamp rebuild happens first, take the tree from `-id` and skip the 5.5% entirely.

**Why it may matter more than a feature — the PRESET STREAM IS DEGENERATE FOR MOST USERS**
(800-user sample, all owned decks): median user has **56 decks, 6 root decks, 1 preset**;
**67.4% of users have exactly ONE preset**, i.e. for two thirds of users the preset stream
pools exactly what the user/global stream already pools. The tree gives a genuine middle
level for **76.5%** of users (decks > roots > 1) and is finer than presets for **92.5%**.
Candidate research moves (both break invariants → Andrew's call):
- **parent/root-deck ID code** as a new input dim group (12 dims like the other IDs; codes
  are re-randomized per batch, identity carried by matching — same machinery).
- **A parent-deck STREAM** replacing or inserted before `preset_id` in the chain
  (card→note→deck→**parent-deck**→preset→global). Note A12 showed preset depth 3L→2L still
  costs accuracy (imm 1.23× the bar), so the preset stack is doing real work even when
  degenerate as a partition — plausibly acting as a second global stream at a different
  time constant. So *augment* looks safer than *replace*; measure both.

### ARBITRARY-DEPTH deck trees — design sketch (Andrew's question, 2026-07-24)

Andrew: "instead of card→note→deck→parent-deck→preset→global, can RWKV process deck trees
of arbitrary depth?" Yes — by making depth a LOOP COUNT rather than an architecture
constant. **Iterative coarsening with weight sharing:** run ONE deck module L times; at
iteration ℓ the rows are grouped by their ancestor ℓ hops up, chained as usual
(x ← module output), with rows that have no ancestor at that level BYPASSING via a mask
(`x = where(active_ℓ, module(x), x)` — the same trick the RNN-baseline probe uses). Depth
becomes data, not parameters.

Why it fits: the 5 streams already ARE "partition rows by entity → run a sequence model per
entity → chain fine→coarse". A level is just another partition, so the WKV kernel is
untouched — only the gathers change (preprocessing emits L sets of deck-style
`sub_gather`/`split_len`/`perm` artifacts instead of 1). States key on (deck, level) — a
deck that is a leaf for some cards and an ancestor for others gets separate states, which
is correct because the input representation differs per level. Add a tiny per-level
embedding (L×d params) so shared weights can still specialize.

**Measured cost (review-weighted, 80-user sample, 7.94M reviews —
`scratchpad/dataset_id/deck_depth_by_review.py`):** 49.3% of reviews sit in top-level decks
(no ancestors); reviews having an ancestor at level 1/2/3/4 = 50.7/42.2/22.9/12.5%, and
96.1% of reviews are at depth ≤ 4. Because inactive rows bypass, an L=4 loop costs
0.507+0.422+0.229+0.125 ≈ **1.28× one deck-stream pass, at ZERO extra parameters** — not
4×. Ancestor entities per level shrink 2,467 → 630 → 265 → 113, so each level is a real
coarsening (a proper pyramid), and extra state ≈ +45 deck-sized states/user (deck states
are the cheap tier; card/note dominate deploy memory).

Variants worth measuring: (a) PARALLEL instead of chained — run all levels off the same
input and sum with level embeddings; order-invariant in depth and all levels compute
concurrently (better GPU utilization), but a bigger break from the chain invariant;
(b) read-only ancestor pooling (single per-deck state, read all ancestors at each review) —
cheaper reads but ancestors never accumulate subtree history unless updates also scatter to
them, which costs the same as (a); (c) depth + ancestor-ID codes as plain FEATURES, no new
stream — the near-free control.

Costs/risks, honestly: needs `parent_id` through preprocessing → **LMDB rebuild** (breaks
the no-new-inputs invariant) and changes the fixed-hierarchy invariant — both Andrew's call.
Deploy needs the Rust engine to loop over ancestor levels with (deck, level) states (Anki
has the real tree at inference, so nothing is blocked). Half the reviews have no ancestor at
all, so any gain must come from the deep-tree half.

**Sequencing (cheap-first, and one rebuild serves everything):** the expensive step is the
LMDB rebuild, so emit gathers for levels 1..6 in that SINGLE rebuild and gate levels by env.
Then: (1) control = fixed 6th stream at the immediate parent (no loop) — if null, the family
is likely null; (2) shared-weight loop L=2/3/4 + level embedding; (3) parallel-pooling
variant; (4) features-only control. The rebuild is CPU-side and can overlap GPU runs
(it does compete with fetch workers for cores).

## Leakage rule
All count/batch features must be computed **as of review time** during preprocessing (not from
the full table) so same-day-created-and-reviewed cards stay honest.

## ★ IMPLEMENTATION PLAN (2026-07-27) — and the delete is probably NOT needed

### ★ DIRECTIVE (Andrew 2026-08-09): the rebuild DROPS Anki's card state from the inputs
The card-state column (new/learning/review/relearning — INPUT_FEATURES.md row 16, the single
`state − 2` column, flat dim 22) must NOT be in the rebuilt feature vector. This is the
permanent version of the decision already in force: iter 15 accepted `RWKV_ZERO_FEATURES=22`
(Andrew's directive), the deploy contract zeroes the dim everywhere, and the Rust export bakes
the zeroed columns into the shipped weights — the rebuild simply stops emitting the column at
the source, so no consumer needs to know a mask ever existed.
Two implementation cautions:
1. **Drop it from the INPUT vector only.** The state column may still be read internally by
   the filtering/label machinery (`create_features`' outlier/continuity filter and the
   equalize selection were built with the benchmark's `--short --secs` settings) — removing it
   from `CARD_FEATURE_COLUMNS` is correct; removing it from the *frame* before the filters run
   is NOT verified safe. Same pattern as `review_time`: derive/filter, then drop before the
   partition assert.
2. **Renumber `RWKV_ZERO_FEATURES` consumers.** With the column gone the vector shifts by one
   at and after dim 22; the mask env (22), Rust `model.rs::load`'s zeroed-column list, and any
   hardcoded dim indices (`COL_DUR=8` is BELOW 22 and safe) must be re-audited against the new
   layout. Grep targets: `ZERO_FEATURES`, `feat_mask`, `COL_`, `dim 22`.

### The four code sites
The RWKV feature vector is built in `rwkv/data_processing.py`, not in `features/` — that is where
every change lands:

1. **`CARD_FEATURE_COLUMNS`** (`data_processing.py:16-41`, currently 24 named columns) — append the
   new names. This list *is* the per-review feature order; the ID encodings make up the rest of the
   92 dims.
2. **`STATISTICS`** (`:43+`) — each new continuous column needs a mean/std, in the same
   `scale_*`/`base_transform_*` idiom already used for `elapsed_seconds` etc. Compute them once on a
   user sample and hardcode, matching what is there now.
3. **`add_segment_features`** (`:185`) — the actual derivations. It already receives the per-user
   revlog frame sorted by review order and asserts `day_offset` monotonicity, so the running
   circular mean (2 floats of state) and the "as of review time" counts are plain cumulative ops
   here. This is the only site where the leakage rule can be got wrong.
4. **The dataset root** (`:196-202`, `read_parquet` of `revlogs`/`cards`/`decks`) — point at
   `anki-revlogs-10k-id`. **Schema-verified 2026-07-27**: `-id`'s `revlogs` is the published schema
   **plus `review_time`**, and `cards`/`decks` are column-identical. ~~So this is a path change, not a
   reader change — every existing derivation keeps working untouched.~~
   ⚠ **CORRECTED 2026-08-03 BY ACTUALLY RUNNING IT — it is NOT a path change, and the schema check
   could not have caught this.** A 20-user probe with only `DATA_PATH` repointed dies at
   `data_processing.py:408`: `AssertionError: review_time not found`. That line is an **exhaustive
   partition assert** — every column of `section_df` must appear in `keep_columns` or
   `reject_columns`, backed by a length assert whose message is *"Ensure that all columns are
   explicitly listed"*. The schema check asked "are the needed columns present?" (yes); the pipeline
   asks "is every column accounted for?" (no). An EXTRA column is exactly the case the first question
   cannot fail on and the second must.
   This is a good assert, and the decision it forces is real: `keep_columns` survive onto the query
   ("no press yet") rows, `reject_columns` are zeroed there because they would leak the outcome.
   **Recommended handling: derive the new features from `review_time` and then DROP it before that
   block** (`df.drop(columns=["review_time"])`). The assert then passes untouched, and a raw epoch-ms
   value — which must never be an input on magnitude grounds alone — cannot leak in by accident.
   Listing it in `keep_columns` would work too, but leaves a 1.7e12-magnitude column one mistake away
   from the feature vector.

### Disk: build on F:, side by side — do NOT delete first
CLAUDE.md records the sequencing as "delete the only copy, no rollback", from the estimate that the
rebuild must land on C:. Measured 2026-07-27, it does not have to:

| | C: (242.2 GB free) | F: (889.5 GB free) |
|---|---|---|
| train | **`train_db_5k_h1` 372.5 GB** (every live run) | `train_db_5k_h2` 372.5 GB (not referenced by any live toml) |
| eval | `label_filter_db` 37.3, `test_db` 22.8, `train_db_sc8k` 3.7, `train_db_sc8k_1500` 74.5 | **`test_db_5k` 232.8 GB** (every live eval) |

A rebuild written to **F:** costs 372.5 (train) + 232.8 (test) = **605 GB against 889.5 free**, so
both new DBs fit **beside** the originals with ~284 GB spare. The old DBs stay readable the whole
time, which means a bad rebuild is a `rm` of the new dir instead of a 2-4 day re-run of the old one.
Only the LMDB_PATH values change (`train_db_5k_h1` is a bare relative path today, i.e. repo root on
C:; `test_db_5k` is already absolute on F:).

⚠ **The test DB must be rebuilt too, not just the train DB** — eval feeds the model the same feature
vector, so a train-only rebuild would silently score a mismatched input layout. Budget both.

If F: gets tight, the honest candidates to reclaim are `train_db_5k_h2` (372.5 GB, the train/eval
swap half, unused since the 5k phase fixed h1) and the closed-era `train_db_sc8k*` + `test_db`
(101 GB on C:). **Both are Andrew's call and neither is needed to start** — flagging, not deleting.

### De-risk before committing anything
Build a **100-user** LMDB from `-id` first (~7.5 GB at the measured ~75 MB/user, trivial on either
drive) and check two things that catch a broken pipeline for ~1% of the cost:
1. ~~**`size` parity** — per-user equalized review count must be unchanged vs the current DB. Row
   counts are already known identical user-for-user and `day_offset` differs on only 4 of 363,598
   reviews (0.001%), so any real movement here is a bug in the new derivations, not the data.~~
   ⚠ **THIS CHECK IS INVALID — measured 2026-08-03; see "`label_filter_db` MUST be rebuilt" above.**
   `size` moves for ~30% of users purely from the dataset swap, so this would false-alarm on a third
   of them. Use the **`-id`-vs-`-id`, new-columns-on vs new-columns-off** comparison instead, where a
   difference is unambiguously the new derivations.
2. **A champion re-run reproduces** on those users with the new columns zeroed/excluded — proving
   the rebuild is additive before any candidate is judged on it.

Then re-base: the champion re-runs on the new DBs and every later candidate is scored against
*that*, since cross-rebuild numbers are not comparable.

---

## ★ THE `STATISTICS` CONSTANTS ARE MEASURED (2026-08-03) — code site 2 is no longer an unknown

Tool: **`optimization/feature_stats_id.py`** (CPU-only, read-only on the dataset, ~45 min).
Sample = **300 users, 24,306,799 reviews**, stride-16 over the **TRAIN half 1-5000 only** — deriving
normalization constants from 5001-10000 would leak eval-set statistics into every candidate's inputs.
Full output: `optimization/feature_stats_id.json`.

### ⚠⚠ THE LANDMINE: the `-id` rebuild introduces NaN features on **3.2% of users**
This is the single most important thing on this page and it would not have surfaced until after a
2-4 day rebuild. `build_parquet_id.py` recomputes `elapsed_seconds` from the corrected SHOW time
(`review_time = revlog.id - taken_millis`). When a review's duration overlaps the next review, the
recomputed gap goes **negative but not `-1`** — and `scale_elapsed_seconds` is

    np.where(x == -1, 0, np.log(1 + 1e-5 + x))

so `x = -26` takes the log branch: `log(-25.99999)` = **NaN**. The NaN flows into the feature vector,
the loss, and then the eval NaN guard skips the whole user.

| | published `anki-revlogs-10k` | `anki-revlogs-10k-id` |
|---|---|---|
| user 486 (11,469 rows, identical count) | **0** bad rows | **1** bad row (`elapsed_seconds = -26`) |
| 313-user sweep, 25,063,241 rows | — | **48 rows = 0.000192%** |
| **users affected** | — | **10 of 313 = 3.2%** (most negative seen: −144) |

Rows are a rounding error; **users are not** — one bad row poisons a user. Projected onto the real
split that is **~160 of 5,000 train users and ~80 of 2,500 eval users**. It would have shown up as
`nan_users` jumping 0 → ~80 *and* a `size` gate failure, with no obvious cause, weeks after the fact.
**FIX (one line, in the derivation, before any log):** clamp to the sentinel —
`elapsed_seconds = np.where(elapsed_seconds < -1, -1, elapsed_seconds)` (same for `elapsed_days`) —
i.e. treat a negative gap as "no previous review" rather than as a magnitude. Decide it deliberately;
silently clamping to 0 instead would claim the two reviews were simultaneous.

### The self-check: it half-passed, and the half that failed says the sample differs
The script recomputes the **nine existing** constants in the same pass, because their derivation was
never written down. Result — `missing->0` is the convention (all four sentinel-bearing columns are
closer under it, decisively so for `elapsed_seconds`: **9.8968 vs upstream 9.96**, where present-only
gives 11.37):

| constant | upstream | recomputed (best convention) | off by |
|---|---|---|---|
| `cum_new_cards_today_mean` | 2.55 | 2.5577 | **0.3%** |
| `elapsed_seconds_mean` | 9.96 | 9.8968 | **0.6%** |
| `duration_mean` | 8.90 | 8.7278 | 1.9% |
| `elapsed_days_mean` | 1.51 | 1.4610 | 3.2% |
| `elapsed_seconds_cumulative_mean` | 10.86 | 11.7164 | 7.9% |
| `cum_reviews_today_mean` | 4.59 | 5.0535 | 10.1% |
| `elapsed_days_cumulative_mean` | 2.14 | 2.4905 | 16.4% |
| `diff_new_cards_mean` | 2.945 | 3.5907 | **21.9%** |
| `diff_reviews_mean` | 4.64 | 5.7411 | **23.7%** |

The pattern is not random: **every badly-missed constant is one that scales with how much the user
reviews** (`diff_reviews` = reviews on other cards between this card's two reviews; `cum_reviews_today`;
`diff_new_cards`), and all three come out **higher** than upstream. So upstream's sample was of
**smaller users** than a stride sample of the train half. Which users, and whether it was drawn before
or after segmentation, is not recoverable from the code.

**=> RECOMMENDATION: do NOT touch the existing nine.** They are part of the model as trained, the
mismatch is a sample difference rather than an error, and each column is standardized to roughly unit
scale anyway — which is all the input FC needs. Use the measured constants **for the new columns
only**, and accept that the new columns are centred on a slightly different sample than the old 24.
(The alternative — recompute all 33 on one sample — is *free at rebuild time* since the rebuild
re-bases everything regardless, but it changes 24 working inputs to fix a cosmetic inconsistency.
Not worth the risk without a reason.)

### The constants (present-only; set undefined rows to log-value 0, the upstream sentinel pattern)

    "t_since_any_review_mean": 2.2682,          "t_since_any_review_std": 1.3649,
    "user_tenure_mean": 17.8880,                "user_tenure_std": 1.1326,
    "creation_to_first_review_mean": 14.8600,   "creation_to_first_review_std": 4.3134,
    "deck_age_at_review_mean": 14.3750,         "deck_age_at_review_std": 5.5264,
    "creation_batch_1min_mean": 2.5430,         "creation_batch_1min_std": 2.1196,
    "creation_batch_1h_mean": 3.6674,           "creation_batch_1h_std": 2.1707,
    "creation_batch_1d_mean": 4.4760,           "creation_batch_1d_std": 2.2149,
    "creation_batch_pos_1h_mean": 3.0207,       "creation_batch_pos_1h_std": 1.9252,
    "deck_depth_mean": 1.2703,                  "deck_depth_std": 1.6451,

Transforms: `log(1 + 1e-5 + x)` for the seconds/duration family, `log(3 + x)` for the count family
(`creation_batch_*`), raw for `deck_depth` — matching the idioms already in `data_processing.py`.
`creation_to_first_review` is **0.2% negative** (a card whose first review predates its own creation
stamp — a re-created card); clamp at 0 rather than adding a sign dim for 1 row in 500.

### Three design corrections the measurement forces

1. **★ Drop `card_id − deck_id` as a single column — it is 57.2% negative.** At n=300 the signed-log
   encoding gives mean −4.59 with **std 15.95**, i.e. a bimodal column whose sign carries most of the
   variance. (The 6-user smoke said 15% negative and looked survivable; it was a small-sample
   artifact.) Cards move between decks constantly, so "card older than its deck" is the *normal* case,
   not an anomaly. **Use `deck_age_at_review` (always ≥0, mean 14.38 / std 5.53) plus a binary
   `card_predates_deck` flag.** Same information, well-conditioned.
2. **Coverage per REVIEW ROW is much lower than the per-deck-row figures above** — that table measured
   the `decks` table, but the model sees review rows, and many rows have a NaN `note_id`/`deck_id`
   merge or the default-deck sentinel:

   | id | per deck row (earlier) | **per review row (what matters)** |
   |---|---|---|
   | `card_id` | 100% | **99.86%** |
   | `note_id` | — | **74.47%** |
   | `deck_id` | 99.5% | **70.22%** |
   | `preset_id` | 7.0% | **18.87%** |

   So deck-derived features are undefined for **~30% of rows**, not 0.5%. `deck_id_is_nan` already
   flags the NaN part but **not** the default-deck (`id == 1`) part → **add an `is_default_deck`
   flag**, or the net cannot distinguish "no deck age" from "deck age 0".
3. **Preset age is confirmed dead; ship only the flag.** Folding its 81% undefined rows to 0 gives
   mean 2.62 / std 6.25 — a column that is one constant 81% of the time. The earlier verdict
   ("treat preset AGE as a low-value add-on, not a peer of deck age") is upheld at 10x the sample.

### The rebuild wall-clock is measured: ~1 day, not 2-4
20-user probe, `PROCESSES = 6`, **while GPU training was running** (so the box was contended):
913,958 reviews in **137 s = 6,671 reviews/s**.

| DB | reviews | projected |
|---|---|---|
| train half 1-5000 | ~372 M | **15.5 h** |
| test half (`test_db_5k`) | ~186 M | **7.7 h** |
| **both** | | **~23 h** |

**This is an OVERestimate, in three independent ways**, which is the direction to want: the probe's
users average 45,697 reviews against the 300-user sample's 81,022, so per-user fixed costs (parquet
open, merges, LMDB txn) amortize over 1.77x fewer reviews than they will in the real run; the box was
sharing CPU with a training run; and `PROCESSES` can go above 6 on a 16-core part when nothing else is
competing. Pushing the other way, the real rebuild derives the new columns too. Net: **plan for about
a day, not the 2-4 days this page previously assumed** — which materially changes when it can be
scheduled (it fits inside one overnight-plus, not a long weekend).
⚠ **NOT included: `find_equalize_test_reviews`** (the 37.3 GB `label_filter_db` helper) — and it
**must** be rebuilt, see immediately below.

### ★★ `label_filter_db` MUST be rebuilt, and the `size` gate WILL legitimately move (measured 2026-08-03)
Tool: `scratchpad/probe_id/check_equalize_drift.py` (read-only, seconds per user). It runs the real
`create_features` on both datasets and reproduces `find_equalize_test_reviews.process()`'s selection —
`TimeSeriesSplit(n_splits=5)` over the surviving frame — then diffs the chosen `review_th` lists.

**40 users, 1-40:**

| outcome | users |
|---|---|
| equalized set IDENTICAL | 12 / 40 (30%) |
| **equalized set DIFFERS** | **28 / 40 (70%)** |
| ... of which **`size` itself changes** | **12 / 40 (30%)** |

Examples: user 17 `size` 108,870 → **109,025**; user 8 45,235 → **45,310**; user 6 65,805 → 65,800.
Deltas are small (±5 … ±155) but they are not zero.

**Why the earlier "0.001% of reviews" reading did not predict this.** That figure was about RAW rows,
and it was right — raw counts are identical user-for-user and `day_offset` moves on 4 of 363,598
reviews. But `create_features` applies **outlier and non-continuity filtering**, which AMPLIFIES a
tiny input change: user 486 loses one surviving row (8,026 → 8,025), while user 17 **gains 188**
(130,645 → 130,833). And because the split is POSITIONAL, even users whose kept-count is unchanged get
a different selection — user 3 keeps exactly 7,089 rows both ways yet 11 `review_th` values differ,
i.e. rows also **reorder** under the corrected show time.
⚠ **The clamp fix for the negative-`elapsed_seconds` landmine will NOT restore parity.** It plausibly
explains user 486's −1 row, but nothing about user 17's +188. This drift is inherent to having more
accurate timestamps, not a defect to repair — the `-id` benchmark set is simply a *different* (better
grounded) one.

**=> TWO RULE CHANGES, both of which have to be made deliberately:**
1. **Acceptance gate #1 currently says `size` must be "IDENTICAL to champion … any change = a pipeline
   bug".** After the rebuild that is false by construction for ~30% of users. Restate it as
   **identical *within a rebuild generation*** — comparisons across the rebuild boundary are not
   size-comparable, exactly as they are not logloss-comparable.
2. **★ The de-risk step below is INVALIDATED AS WRITTEN and must be redesigned.** It says to check
   "per-user equalized review count must be unchanged vs the current DB … any real movement here is a
   bug in the new derivations, not the data." That check would now fire on ~30% of users *for correct
   reasons* — worse than useless, because it trains the reader to dismiss the one alarm that was
   supposed to catch a real pipeline bug.
   **Replacement that actually discriminates:** build the 100-user probe DB from `-id` **twice** —
   once with the new feature columns enabled and once with them disabled — and require `size` and the
   equalized sets to match **between those two**. Same data source, so any difference is unambiguously
   the new derivations. Separately, and only as an expectation rather than a gate, `-id`-vs-published
   drift should land near the 70% / 30% measured here; wildly more would mean something else moved.

### Time-of-day has real signal (the feature is not degenerate)
Mean resultant length **R = 0.415** (median 0.414, p10 0.197, p90 0.624) over 300 users. R≈0 would
mean users review uniformly round the clock and "deviation from the usual hour" would be dead; R≈0.41
means a clear preferred window with real spread. The p10 of 0.197 says ~10% of users are near-uniform,
which is an argument for **also** feeding the raw phase (the recurrent user stream can then learn the
per-user concentration itself) rather than the deviation alone.

---

## ★★ DECK TREE: THE NO-REBUILD PATH IS CONFIRMED AT SCALE (2026-08-15)

Andrew asked for the real thing: `card->note->deck->preset->global` becomes
`card->note->(deck, depth_level)->preset->global`, i.e. the shared-weight ancestor loop sketched
above, not the fixed-parent control.

**Two contradictory claims in this file are now settled in favour of NO REBUILD.** The design sketch
says the tree "needs `parent_id` through preprocessing -> LMDB rebuild"; the 2026-07-26 correction
says it is "usable on the CURRENT LMDBs, unmodified". The correction is right, and the reason is
structural rather than empirical: `data_processing.get_rwkv_data` drops `parent_id` (:228) but
**never factorizes or remaps `deck_id`** -- the only rewrite is NaN -> `ID_PLACEHOLDER` (:243-245).
So the parquet's own `deck_id -> parent_id` mapping applies directly to ids already in the LMDB.
The sketch's rebuild line is superseded; it was written before that was checked.

**Measured, `scratchpad/deck_tree/` (40 users, ALL chunks, 3.67 M reviews):**

| | distinct deck ids | reviews |
|---|---|---|
| no deck row (deleted/filtered; `df_decks` merges how="left") | 0.65% | 17.21% |
| known root, no parent | — | 33.58% |
| **has an ancestor** | **95.26%** | **49.21%** |

**49.21% review-weighted independently corroborates the design sketch's 50.7%**, which was measured
by a different route entirely (`deck_depth_by_review.py`, 80 users, on the `-id` dataset). Agreement
to 1.5pp across two unrelated measurements is the confirmation the one-user 14/14 spot check could
not give. Distinct-id resolve (95.26%) likewise matches the recorded 94.5%.

**⚠ AND A SAMPLING TRAP, caught before it became a wrong conclusion.** The first pass read only
`keys[0]` -- each user's EARLIEST chunk -- and reported **34.16%** review-weighted with 3.81% of
distinct ids unknown. Both are artifacts: a user's earliest decks are the most likely to have been
deleted since, so the earliest chunk maximally over-represents deck-row-less rows. Sampling across
all chunks moved distinct-id-unknown 3.81% -> 0.65% and reach 34.16% -> 49.21%. Reporting the first
number would have said the tree reaches 1.5x less of the corpus than the sketch predicted, and would
have argued for dropping the idea. **Same family as the control/metric traps of 2026-08-14: the
sample has to span the axis you are integrating over.**

**Rows with no deck row are NOT a correctness problem** -- they bypass exactly like roots. They only
bound reach, and the bound is already priced in: ~half the corpus can move, which is what the sketch
assumed.

Tools: `build_parent_maps.py` (emits `(user_id, deck_id, parent_id)`, `-1` = no resolvable parent;
0.1 MB per 60 users, with a cycle guard) and `verify_lmdb_link.py` (the LMDB-side check above).
