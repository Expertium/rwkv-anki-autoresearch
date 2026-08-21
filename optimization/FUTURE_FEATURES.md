# ★★ GENERATION-2 REBUILD (Andrew 2026-08-20): "we still have to do another LMDB rebuild"

> *"Let's do bundle first, but we still have to do another LMDB rebuild"*

**✓ DONE 2026-08-20 21:15:05, `DONE_EXIT_0`, 3 h 57 m** (17:18 -> 21:15), entirely inside featA's
runtime and therefore at zero GPU cost.

| db | entries | width | actual GiB |
|---|---|---|---|
| `train_db_5k_h1_id2` | **1,483,984** (identical to gen 1) | 46 | 118.3 |
| `test_db_5k_id2` | **170,384** (identical to gen 1) | 46 | 118.5 |

Both entry counts match generation 1 exactly, which is the integrity check: chunking is unchanged
and the added columns are per-review, so a DIFFERENT count would have meant row filtering had moved.
Gen 2 costs +2.2% of disk for +4.5% of columns, because the 68 ID-encoding dims are unchanged.
Generation 1 is KEPT as a verified fallback -- F: has 414 GiB free, so reclaiming its 231.7 GiB buys
nothing.

**Both omissions are implemented.** `NEW_COLUMNS` goes 21 -> 23,
card_features 44 -> 46, model input 112 -> **114**, params 558,212 -> **565,252** (verified by
constructing the model, not by arithmetic). New dbs `train_db_5k_h1_id2` / `test_db_5k_id2`;
`label_filter_db_id` is REUSED because it selects which reviews count, not what they contain.

### Why this cost no GPU time
featA reads the OLD 92-dim dbs and the `_id` dbs had no reader, so a CPU-only rebuild ran fully
inside featA's own runtime. Doing it in this window means featB measures the COMPLETE 23-column
bundle instead of a 21-column one we would then have had to re-measure -- about 11 h saved against
the alternative ordering (run featB on 21, rebuild, run a third arm on 23).

### ★ THE DISK PLAN WAS WRONG BY 2.5x, IN THE SAFE DIRECTION
This page budgeted "605 GB against 889 GB free". That is the **logical** size: LMDB map_size is
sparse on Windows and the file length equals the map, not the allocation. Measured with
`GetCompressedFileSize`, generation 1 actually occupies **115.8 GiB (train) + 115.9 (test)**, not
372.5 + 232.8. The tell arrived by accident -- deleting two finished 27.9 GiB de-risk dbs returned
**3 GiB**, not 55.8.
**The rule: for any sparse-allocating store, `Get-ChildItem | Measure Length` reports the RESERVATION
and free-space arithmetic built on it is fiction.** Use `GetCompressedFileSize`, or measure free
space before and after.

### ★ COVERAGE MEASURED FIRST, AND IT LOWERS THE PRIOR ON THE SIBLING GAP
`scratchpad/features_rebuild/sibling_stats.py`, stride sample of the TRAIN half only:

| quantity | coverage |
|---|---|
| rows with a preceding sibling review (**the gap**) | **~10-16%** |
| rows whose note had >=1 OTHER card created earlier (**the ceiling**) | **~17%** |

For scale: `preset_age` was DROPPED from generation 1 for being defined on 1 row in 14 (7.1%), and
the deck-tree parent level reached **49.2%** and returned an exact tie (iter 50). The sibling gap
sits between them and much nearer the dropped one. **Pre-registered expectation: this column is
unlikely to move the gate.** It ships anyway because Andrew asked for it and because of the
asymmetry below.

### ★ THE ASYMMETRY THAT DECIDED WHAT SHIPS
**A column that is IN the db can be ablated without rebuilding; a column that is OUT costs a full
rebuild to add.** So the cheap error is to include a column that turns out dead, and the expensive
error is to omit one. That is why the low-coverage sibling gap is in, and why the redundancy screen
is now evidence for INTERPRETING the result rather than a gate on shipping it.

⚠ **AND THE ABLATION MECHANISM DOES NOT EXIST YET -- checked, not assumed.** `RWKV_ZERO_FEATURES` is
**hard-refused** whenever `RWKV_ID_FEATURES=1` (`srs_model.py:438`, mirrored at
`srs_model_rnn.py:63`). The refusal is CORRECT on its own terms: the rebuild drops the card-state
column at the source, so the historical `=22` would silently mask `day_of_week` instead. But it also
means the family-level ablation plan below is **not executable today**.
**OWED, and it is small:** a NAME-based `RWKV_ABLATE_FEATURES=scaled_sibling_gap,tod_sin` resolved
through `CARD_FEATURE_COLUMNS`, which is immune to the renumbering the numeric guard exists to
prevent. It must land in **both** `srs_model.py` and `srs_model_rnn.py` (three-way parity) with a
smoke case, and the numeric refusal stays.
**⚠ DO NOT WRITE IT WHILE A CHAIN IS MID-FLIGHT.** A chain's later phases are new processes that
import whatever is on disk then, and a **plain eval is the only path that TorchScript-compiles the
model** -- which is exactly how iter 48 lost an eval to a `@torch.jit.ignore` return-type bug that
the compile-side check had passed. Land it when no eval is pending, and run
`scratchpad/parity3/smoke_scripted_eval.sh` after.

### ★ A THIRD COLUMN WAS CONSIDERED AND REJECTED BY A PRE-REGISTERED RULE
`scaled_sibling_count` (how many siblings existed as of review time) is the high-coverage half of
Andrew's interference intuition and the note stream plausibly does not encode it. The rule was
written down BEFORE the number: ship it only if >=30% of rows have >=1 prior sibling card. Measured
**17%**. It does not ship. Recording the threshold in advance is what makes this a decision rather
than a rationalisation.

### The non-leaking form, restated because the code now depends on it
Andrew's literal `min(|t_now - t_sib|)` over ALL siblings includes FUTURE sibling reviews. Restricted
to preceding siblings it collapses to `t_now - max(t_sib)`, i.e. plain recency with no `min`.
`id_features.sibling_gap_seconds()` implements that with a block trick (sort by note+time; adjacent
blocks differ by card, so the answer for every row of block b is the last row of block b-1), which
is O(n log n) instead of a per-row sibling scan over 372 M rows. END-to-START like every other
interval here, clipped at 0, sentinel -1 where there is no preceding sibling.

### ★ THE BLOCK TRICK IS VERIFIED AGAINST BRUTE FORCE, NON-VACUOUSLY
`scratchpad/features_rebuild/verify_sibling_gap.py` recomputes the gap the obvious O(n^2) way --
each row scans its own prefix for the latest different-card review of the same note -- and compares.
User 101, **37,122 rows, 3,035 defined, 2,986 nonzero: max |delta| = 0.000e+00.**

⚠ **THE FIRST VERSION OF THAT CHECK WAS VACUOUS AND SAID `0.000e+00` ANYWAY.** It ran on user 1,
who has **4005 cards and 4005 notes -- not one multi-card note** -- so it compared an all-sentinel
array against an all-sentinel array. The file now REFUSES to report agreement unless the reference
contains >200 defined rows and >100 nonzero gaps. Same family as the parity harness's rule about
randomizing zero-init params before comparing: **an agreement between two empty results is not
evidence, and it looks exactly like success.**
⚠ A second self-inflicted one, the same ten minutes: an unchecked `str.replace()` silently failed to
match, left `df.head(0)` in place, and produced a "contradiction" between brute force and the block
trick that was entirely my own harness. **Assert the match count on every programmatic edit** -- the
repo already learned this for runner generators; it applies to throwaway scripts too.

### ★ PER-USER COVERAGE IS FAR MORE SKEWED THAN THE AGGREGATE (`sibling_value_check.py`)

| user | reviews | sibling gap defined | card_predates_first_review = 1 |
|---|---|---|---|
| 1 | 22,430 | **0.00%** (4005 cards / 4005 notes) | 0.14% |
| 2 | 69,765 | **0.41%** | **49.94%** |
| 101 | 37,122 | **8.18%** | 17.50% |

Two readings, and they point opposite ways:
* **The sibling gap is near-dead for most users.** User 2 has 53% of NOTES carrying >1 card yet only
  **2.03% of REVIEWS** land on them -- multi-card notes exist and are barely studied. The ~10-16%
  aggregate is carried by a minority of users; the per-user p10 is 0.0000.
  Where it IS defined the shape is bimodal and sensible: user 101's median gap is **0.00 days**
  (p90 ~30 min -- burial off, same session) while user 2's is **118 days** (p90 2.8 years -- two
  cards of one note that simply never met). Only the first mode is "sibling interference".
* **`card_predates_first_review` is the stronger of the two omissions, and that was not expected**
  (Andrew filed it `low`, "probably not important"). It ranges 0.14% -> 49.94% across three users,
  i.e. it separates a user who imported a collection wholesale from one who built it while studying.
  That is a user-level property the model currently has no column for.

### ★ GEN 1 AND GEN 2 AGREE EXACTLY ON ROW STRUCTURE (checked 2026-08-20 20:20)
`check_db` on both train dbs: **entries = 1,483,984 in BOTH**, width 44 vs 46. Chunking is unchanged
(`MAX_BATCH_SIZE = 16384`) and the two added columns are per-review values, so an identical count is
what "additive in columns" has to mean -- and a DIFFERENT count would have been the cheap signal
that something in row filtering had moved. This makes gen 1 -> gen 2 a genuine single-variable
change. Cost of the check: one `txn.stat()` per db, which is metadata, not a scan.

★ The priority loop also earned itself here: phase 2 spawned **7 fresh workers at Normal priority**
at 20:18:48 and the loop reniced them within 30 s. A one-shot renice would have lapsed silently at
exactly the phase boundary -- which is the phase whose generation-1 counterpart OOM'd.

### ★★ THE REDUNDANCY SCREEN RAN (2026-08-20) -- THE NOTE STATE DOES CARRY SIBLING RECENCY

The item owed from Andrew's own 2026-08-18 filing. `scratchpad/features_rebuild/
sibling_redundancy_screen.py`: walk user 101 through the DEPLOY RNN path with the iter-53 champion,
capture the **INCOMING** note state at each review (what the model holds when it predicts THAT
review), ridge-regress the true sibling gap on it, 70/30 held out.

| regressor | held-out R2 |
|---|---|
| **note state (1,440 dims)** | **+0.4431** |
| SHUFFLED target, same dims | -0.1749  (the overfitting floor) |
| review index (1 dim) | +0.0282  (trivial baseline) |

The state dimension came out at exactly **1,440**, matching the recorded note state size -- a free
cross-check that the right stream was captured.

**READING: a null on `scaled_sibling_gap` is EXPECTED and would be uninformative.** This is iter 50's
shape -- the hierarchy already brackets the information, so the model gains nothing from being handed
it explicitly. It does NOT mean sibling recency is irrelevant to recall; it means the recurrence
already reconstructs most of it.

**✓ CONFIRMED AT SECONDS RESOLUTION -- the shipped quantity, not a coarsening of it.** Same user,
`-id` frame so target and state share a frame: **R2 +0.3068**, floor -0.2188, trivial baseline
+0.0008. Target std 2.706 (vs 0.743 in day mode) and only **1.6%** of targets at exactly zero, so the
sub-day structure really is present and the day-mode number was not an artefact of the collapse.
Lower than day mode's 0.4431, as it should be -- sub-day recency is harder to reconstruct -- but
decisively above the floor.
**=> the note stream reconstructs roughly a third of the variance of the exact column we ship.** The
remaining ~69% is the honest counterweight: the state does NOT fully encode it, so there is room in
principle. Against ~10-16% coverage, the expected value stays low.

⚠ **AND THE DAY-RESOLUTION VERSION UNDER-TESTS THE SHIPPED COLUMN, which is why a seconds-resolution
run followed.** The target above is built from the published set's `day_offset`, and user 101's
MEDIAN gap is **0.00 days** (p90 ~30 min) -- so most targets collapse to zero and the regression is
largely predicting SAME-DAY vs EARLIER. The column we ship is seconds-resolution end-to-start, and
the sub-day structure is exactly where that user's mass lives. The `--id` mode puts the real target
and the state in the same frame; day mode alone would have supported a broader claim than it earns.

### ★★ PER-ROW REDUNDANCY SCREEN (2026-08-21, CPU-only) -- IT SETTLES THE ABLATION GRANULARITY

`scratchpad/features_rebuild/feature_redundancy_screen.py`, 4 users / 60,000 rows, ridge held out
70/30. **R2_old** = predictability of a new column from the 23 ORIGINAL per-row features;
**R2_all** = from the originals PLUS the other 22 new columns. Shuffled floor is ~-0.001, so
anything above ~0.01 is real.

| family | R2_old (new info?) | R2_all (intra-family redundancy) |
|---|---|---|
| time-of-day (4 cols) | **0.08-0.20** | 0.26-0.44 |
| calendar (5) | **0.02-0.12** | 0.11-0.65 |
| recency+ages (3) | 0.12-0.39 | 0.49-0.71 |
| deck (4) | 0.24-0.46, +1 dead | 0.75-0.91 |
| creation-batch (4) | 0.23-**0.66** | **0.83-0.95** |
| preset (1) | **0.67** | 0.92 |
| the two omissions | 0.10 / 0.17 | 0.28 / 0.60 |

**★ THE HEADLINE, AND IT IS A GPU DECISION: R2_all IS 0.6-0.95 FOR MOST COLUMNS, SO PER-FEATURE
ARMS WOULD LARGELY MEASURE SHARED INFORMATION.** Removing ONE column leaves its information sitting
in its siblings, so a single-column ablation is expected to read as null even for a column that
matters. This converts "23 arms is expensive" (7-8 days) into "23 arms is also UNINFORMATIVE", which
is a much stronger reason to ablate by FAMILY. The creation-batch four are the extreme case at
0.83-0.95 -- four columns doing roughly one column's work.

**★ WHAT IS GENUINELY NEW PER-ROW:** time-of-day and calendar, at R2_old 0.02-0.20. That matches
their design rationale exactly -- the existing vector has only `day_offset`-derived PSEUDO-phase
cycles, so true wall-clock phase was unavailable. `dow_cos` is the most novel single column in the
whole set at 0.022.

**★ WHAT IS LARGELY ALREADY THERE:** `is_default_preset` (0.67) and `scaled_creation_batch_1d`
(0.66) are two thirds reconstructible from the current row alone, and `scaled_t_since_any_review`
is 0.39 -- unsurprising, since `scaled_elapsed_seconds` is already an input and the two differ only
by "this card" vs "any card".

**⚠ `is_default_deck` IS NEAR-CONSTANT (std 0.006) on these users and its R2 is UNDEFINED, not bad.**
A column with no variance carries nothing whatever its redundancy. ⚠ Four users is too few to call
it dead globally -- default-deck usage is a per-user habit -- so this needs a wider count before
dropping it. The first version of the screen printed **R2 = -2.0e9** here; the tool now refuses
rather than clamps, because a clamped -1e9 still reads as a value.

**WHAT THIS SCREEN CANNOT SEE, so it is not over-read:** it is PER-ROW. It cannot tell whether the
RECURRENCE could derive a column over time from its state -- that needs a state dump, which is what
`sibling_redundancy_screen.py` does for one column and found ~31%. So a LOW R2_old means "not
trivially present in the current row", NOT "unavailable to the model". A HIGH R2_old is the
decisive direction.

### ★★ PRE-REGISTERED: HOW featB - featA GETS READ (written 2026-08-20 20:15, BEFORE any number)

Recording this now because the alternative is deciding the bar after seeing the result, which is
how a null becomes "promising" and a small win becomes "decisive". The sibling-count threshold
earlier today is the precedent: the rule was written first and it cleanly rejected a column I
wanted to like.

**What the comparison is.** featA and featB are the SAME recipe -- iter 53's, minus KD -- differing
only in the db paths and `RWKV_ID_FEATURES`. So `B - A` is the value of the new input pipeline and
nothing else.
**KD is off in BOTH arms and NOT by choice:** the d=128 teacher's `features2card` has `in_dim = 92`
and cannot forward-pass a 114-dim row, so a KD dump cannot be produced for arm B. Comparing arm B
against the KD-ON champion would confound the features with removing KD (~0.0019 across iters
32/35/39/45).
**⚠ NEITHER ARM IS COMPARABLE TO ITER 53.** They are a new KD-off generation. Do not put their
absolute numbers in the champion table, and do not report `featB - iter53` as the features' effect.

**The `size` gate does not apply ACROSS the arms.** The dataset swap alone moves the equalized
review count for ~30% of users, so gate #1 is only meaningful WITHIN an arm (arm B vs a future arm
B'). Check it within-arm; a cross-arm `size` difference is expected, not a pipeline bug.

**⚠⚠ CORRECTION TO THIS PRE-REGISTRATION, found 2026-08-20 22:00 while reading the redundancy
screen's output. `featB - featA` IS NOT "the 23 new features". IT BUNDLES FOUR CHANGES**, and the
paragraph above that called it "the value of the new input pipeline and nothing else" was wrong about
the "nothing else":

1. **the 23 new columns** -- the intended variable;
2. **END-to-START intervals** instead of start-to-start (Andrew 2026-08-19). Verified gated on
   `"review_time" in df.columns` (`data_processing.py:208,279`), so it can ONLY ever exist on the
   `-id` set -- it is inseparable from the swap by construction, not by oversight;
3. **the cumsum sentinel fix** (obezag's report: the -1 first-review sentinel was being summed into
   `elapsed_*_cumulative`). This one is NOT dataset-gated (`:333-337`) -- but the OLD dbs were built
   BEFORE it landed (`9ebf23d`), so they carry the buggy cumulative values baked in and featB's do
   not;
4. **the dataset swap itself**, published -> `-id`.

**Why this does NOT invalidate the comparison, and what it does change.** The decision on the table
is "adopt the new input pipeline?", and the pipeline IS all four -- so `B - A` is still the right
quantity for THAT decision, and the bands below stand. What changes is ATTRIBUTION:
* a NULL would NOT mean "the 23 features are worthless" -- it could equally be features helping while
  something else costs, or the reverse;
* a WIN would not be attributable to the features specifically;
* **item 3 is a BUG FIX and should be adopted whatever the bundle says.** If featB wins partly
  because of it, that is a reason to rebuild the control, not to credit the features.

**A first, weak datapoint on component 4, free from the redundancy screen.** The same champion,
same `RWKV_ID_FEATURES=0`, same user 101, run on published vs `-id`: imm 0.323313 -> 0.323779
(**-0.000466**), ahead 0.357672 -> 0.357775 (**-0.000103**), `size` 23,190 IDENTICAL on both. So the
swap alone looks slightly NEGATIVE, not positive -- but this is ONE user, at inference time only,
with a model trained on published-derived data, so the `-id` inputs are mildly off-distribution for
it. Treat it as "the swap is not a large free win", nothing more.

**If the bundle wins and attribution matters, the clean follow-up is one arm at
`RWKV_ID_FEATURES=0` on the gen-2 dbs** -- that isolates items 2-4 from item 1 for one 7.75 h run,
and it is the arm to spend on before any per-family ablation.

**The bands, fixed in advance (both modes, raw, vs featA, with p < 0.0001):**

| `B - A` | reading | what happens next |
|---|---|---|
| **>= +0.0010 both modes** | large, i.e. ~5-10 accepted iterations in one step | adopt as the trunk; the 7-arm FAMILY ablation is then worth its ~54 h |
| **+0.0003 to +0.0010 both** | real and clears the 7.5e-5 floor decisively | adopt (the rebuild cost is already sunk; what remains is one champion re-base plus the Rust input-width port), but do NOT spend 7 arms -- ablate only the 2-3 families with a mechanism story |
| **< +0.0003, or mixed sign** | the 23 columns as designed do not pay | do NOT adopt; record which families to attack differently, and do not run per-feature arms, which would measure noise 23 times |

The >= 0.0010 band is set by Andrew's 2026-08-19 steer -- *"I don't think that chasing 0.0001 is
worth it... focus on major changes that are likely to have a large impact"* -- against the reference
points that the whole A0 -> A18 width ladder cost +0.00053 imm and iters 32-53 accumulated ~+0.0019
over ten accepted iterations.

**HOW THE TWO ARMS GET LOGGED (decided in advance, 2026-08-21).** Neither arm is a champion
candidate: both are KD-off and neither is comparable to iter 53, so neither can be an accept.
* Log BOTH to `research_log.jsonl` with **`number: null`** and `status` `control` / `treatment`,
  plus one `research_5k_verbose.md` section covering the pair, and rebuild `log.md`.
* Do NOT assign iteration numbers. The numbering convention is "the Nth RESULT in the champion
  lineage", and these two are a phase decision beside it. Assigning numbers would put two non-
  comparable rows into a table whose whole meaning is comparability.
* An iteration number is earned only if the pipeline is ADOPTED and a champion RE-BASE run follows
  on the gen-2 dbs -- that run is the first thing in the lineage again, and it takes the next number
  at its own verdict.

**If featB is WORSE, check these THREE things before believing it.** More input dims at unchanged
trunk capacity can genuinely hurt, but three artefacts look identical to that:
1. the param guard -- `Trainable parameters: 565252` must appear in arm B's WS log (the guard
   already fails the run if not), which is what catches `RWKV_ID_FEATURES` not reaching the workers;
2. within-arm `size` -- if arm B's equalized counts are internally inconsistent, that is a pipeline
   bug and not a result;
3. the eval db -- arm B's `eval.toml` must name `test_db_5k_id2`, since scoring 114-dim weights
   against a 112- or 92-dim db is a silent shape mismatch. The runner greps for it.

**Per-feature testing stays gated on the bundle winning.** 23 single arms is 7-8 days of GPU; the
whole reason for bundle-first is that a null bundle makes those 23 arms measurements of noise.

### Smokes, all green before launch
* `smoke_id_features.py` -- **PREFIX INVARIANCE at exactly 0.000e+00**, and it covers the new
  columns automatically because it iterates `NEW_COLUMNS`. Finiteness, determinism, the
  `create_sample` partition assert (42106, 46), causal-clip checks all pass.
* `smoke_id_features_width.py` -- 92 and 114 agree across the training class, the deploy RNN class
  and `CARD_FEATURE_COLUMNS`.
* Param count 565,252 verified by constructing `SrsRWKV`, and baked into featB's runner guard.

### ⚠ The generator now takes an ARM FILTER
`mk_features_ab.py` rewrites BOTH runners, and featA's was RUNNING. cmd.exe re-reads a batch file
from a saved byte offset, so rewriting a live runner makes it resume mid-garbage -- the trap that
cost iters 43 and 46. `python mk_features_ab.py featB` now regenerates one arm only.

---

# ★★ FEATURE-COVERAGE AUDIT + PER-FEATURE TESTING (Andrew, 2026-08-20)

> *"I was thinking we should try each new feature separately. Also, you forgot the number of days
> since the closest sibling review, check if there are any other features you forgot."*

## What was implemented vs what this page proposed

The rebuild shipped **21** columns. Cross-checking them against this page's own candidate table:

| candidate | priority | in the rebuild? |
|---|---|---|
| time-of-day raw + circular-mean deviation | high | YES (4 cols) |
| real-phase calendar cycles | high | YES (5 cols: dow, doy, is_weekend) |
| first review - card creation | high | YES |
| seconds-resolution time since any review | high | YES |
| creation-batch 1min/1h/1d + position | med | YES (4 cols) |
| user tenure | med | YES |
| deck age at review, is-default, depth, card-predates-deck | med | YES (4 cols) |
| preset age | med | DROPPED ON PURPOSE -- defined for 1 row in 14 |
| `card_id - note_id` gap / session count per day | skip | correctly skipped |
| **SIBLING REVIEW GAP** | **queued** | **NO -- MISSED** |
| **card created before vs after the user's first-ever review** | **low** | **NO -- MISSED** |

### The two genuine omissions

1. **Days since the nearest PRECEDING sibling review** (Andrew's, filed 2026-08-18 with its own
   section below). It was analysed here -- including the correction that a `min` over ALL siblings
   leaks the future, so the deployable form is "time since the most recent sibling review" -- and
   then simply never added to `NEW_COLUMNS`. Filing a feature is not implementing it, and nothing in
   the pipeline notices the difference.
   ⚠ Its **CPU redundancy screen was also never run**: regress the true sibling gap on the champion's
   note-stream state and read the R2. High R2 means the note stream already carries it (the iter-50
   outcome) and it should be dropped; low R2 means it adds information the recurrence cannot derive.
   That screen costs ~90 min of CPU and gates whether it deserves a slot at all.
2. **Card created before vs after the user's first-ever review** -- `low`, Andrew's own "probably not
   important, but we can try". Cheap (one boolean) and never added.

Neither can be added without ANOTHER rebuild, since both are stored columns.

## ★ PER-FEATURE TESTING: the cost, and what to do instead

Andrew asked to try each feature separately. Priced honestly at the measured ~7.75 h per arm
(KD-off, 1 WS epoch + decay + a 2500-user eval):

| granularity | arms | GPU time |
|---|---|---|
| one at a time (21, or 23 with the omissions) | 22-24 | **~170-186 h = 7-8 days** |
| by FAMILY (time-of-day / calendar / recency+ages / deck / creation-batch / preset) | 7 | **~54 h** |
| bundle now, then ablate only what won | 2 + k | 15.5 h + 7.75 h per ablation |

**RECOMMENDATION: bundle first, ablate second** -- which is what the running A/B already does.
The bundle answers the decision that actually gates everything ("is the new pipeline worth
adopting?") in 15.5 h. If it wins, ablating the winners is a targeted follow-up on a *smaller* set.
If it loses, per-feature runs would mostly be measuring noise around zero, 21 times.

⚠ **The one case that argues for going finer FIRST:** a harmful feature can mask a helpful one in a
bundle. That is real, but it is cheaper to detect it *after* a null bundle -- by ablating families,
7 arms not 21 -- than to pay for 21 arms up front on the chance that it happened.

**If Andrew wants finer granularity anyway, FAMILY level is the right unit**: 7 arms, ~54 h, and the
families are already the natural groupings in `NEW_COLUMNS` (they are even commented as such).

# ★★★ STATUS: THE REBUILD IS DONE (2026-08-20 01:27). Everything below is the PLAN it was built from.

Andrew authorized it 2026-08-19; it completed in **~4 hours of wall clock**, not the ~23 h this page
estimated. That estimate was already flagged here as unmeasured at scale, and it was wrong by ~6x in
the helpful direction.

| artifact | path (all on F:, originals untouched) | verified |
|---|---|---|
| label filter | `F:/rwkv_lmdb/label_filter_db_id` | 20,000 entries |
| train, users 1-5000 | `F:/rwkv_lmdb/train_db_5k_h1_id` | 1,483,984 entries, **card_features width 44**, **0 users missing** |
| eval, users 5001-10000 | `F:/rwkv_lmdb/test_db_5k_id` | **card_features width 44**, **0 users missing** |

Width 44 = 24 original columns - 1 (Anki card state, dropped per Andrew 2026-08-09) + 21 new
real-timestamp columns, i.e. the model input goes **92 -> 112**.

**TWO CORRECTNESS FIXES ARE BAKED IN**, both found and fixed during the build and neither part of
the original plan:
1. **The -1 sentinel was being SUMMED into `elapsed_*_cumulative`** (reported by obezag on Discord).
   Because the feature is `log(1+x)`, storing `C-1` gave `log(C)` where `log(1+C)` was meant -- and
   at C=1 that is exactly the value the sentinel encodes, so 3.9% of non-first reviews were
   indistinguishable from "no history".
2. **Intervals were START-to-START, not END-to-START** (Andrew). The `-id` set corrected timestamps
   to SHOW time, which is why it looked done, but `review_time.diff()` still carried the previous
   review's duration. Fixed for `elapsed_seconds` (per card) and `t_since_any_review` (per user);
   deliberately NOT for `elapsed_days` (a calendar-day index) or the age features (already anchored
   to show time). Worth 31.6% on sub-minute gaps, ~0% beyond a day.

**OPS LESSON FOR ANY FUTURE REBUILD: the WHOLE-USER (eval) config must drop to `PROCESSES=2`.**
Each worker holds an entire user's matrix and those are now 1.8x wider, so the inherited
`PROCESSES=6` exhausted 64 GB and died on a 4 MB allocation. The train config is unaffected -- it
chunks at 16384. (Compounded by a concurrent training run whose four fetch workers measured ~9.7 GB
EACH, against the ~2.6 GB CLAUDE.md records.)

**STILL OWED before a candidate can be judged on these DBs:**
* **re-base the champion** -- re-run it on the new DBs; cross-rebuild numbers are NOT comparable;
* training/eval tomls pointing at the new paths, with `RWKV_ID_FEATURES=1`;
* the **Rust input-width port** (92 -> 112) for the deploy path.

---

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

## ★ SIBLING REVIEW GAP (Andrew, 2026-08-18) — queued, with one correction and one cheap redundancy test

> *"if a card has siblings, then calculate the minimum number of days between sibling reviews.
> Example: note A has cards A1, A2 and A3. A1 was reviewed on day 100, A2 on day 95, A3 on day 110.
> So for card A1, the input feature will be min(abs(100-95), abs(100-110)). Well, with log1p... This
> may be redundant thanks to note states though."*

### ⚠ The example as written LEAKS, and the causal version is simpler than the general one
`A3` is reviewed on day **110**, which is *after* the day-100 review being featurised. Taking a
`min` over all siblings therefore lets the model see the future — not deployable (in Anki, when
scheduling A1 you know your siblings' past reviews and not their future ones) and it would inflate
eval exactly where no gate would catch it, since this is an input-side change and `size` would be
identical.

Restricted to siblings reviewed **before** the current review — which is the only deployable form —
the definition collapses:

    min over PAST siblings of |t_now - t_sib|  ==  t_now - max(t_sib)  ==  time since the most
                                                                          recent sibling review

So the feature is **"days since the nearest preceding sibling review"**, one column, and it needs no
`min` at all — a running per-note "last review time" carried forward, which is cheap in the
preprocessing pass and matches how `elapsed_seconds` is already computed. (Andrew's example still
gives 5, because there the nearest sibling happens to be the past one; the two forms only diverge
when the nearest sibling is in the future, which is precisely the leaking case.)
This is the same shape as the existing **Leakage rule** below — compute as of review time, never
from the full table.

### Is it redundant with the note stream? — TESTABLE ON CPU, BEFORE THE REBUILD
Andrew's own caveat is the right worry, and it has a precedent that cuts both ways: **iter 50 (the
deck tree) was an exact tie because the 5-stream hierarchy already BRACKETED the scope it added.**
The `note_id` stream does pool every review of a note, so the note state has *seen* the sibling
reviews. But seeing them and encoding *the gap to the nearest one* are different claims — the state
would have to reconstruct a time difference through its recurrence.

**The cheap test, and it costs no rebuild:** the champion's note-stream state is already dumpable
(the deploy RNN path gives it per review; `spectra.py`/`calibration_by.py` show the pattern). Regress
the true sibling gap on that state and read the R²:
* high R² → the note stream already carries it → **redundant, drop it** (and that is a real finding
  about what note state encodes, not just a rejection);
* low R² → the recurrence does *not* represent it → the feature adds information the model cannot
  currently derive, and it earns a slot in the rebuild.
Run this the way the spacing and horizon screens were run: **before** committing GPU or a 1-day LMDB
rebuild. Two screens have already killed queue items this week for ~90 min of CPU each.

### Why it is plausible even so
Anki *buries* siblings by default, so the gap is largely schedule-determined — but that is what makes
it informative: a small gap means burial was off or overridden, and sibling interference (two cards
of one note studied close together) is a real memory effect the model currently has no explicit
handle on. Note also that the quantity is **not** symmetric with what the note stream sees: the note
state is dominated by *how many* siblings and *how they went*, not by *how recently*.

### Cost and placement
One extra column (width 112 → 113 under `RWKV_ID_FEATURES`), derivable from the `-id` set's real
timestamps, no new export. It belongs to the **features phase**, i.e. the same LMDB rebuild as the
other candidates — it does not justify a rebuild on its own.

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

**★ AND THE TREE IS DEEP -- the depth histogram PEAKS AT 4, not 1** (`scratchpad/deck_tree/level_reach.py`,
same 40 users / 3.67 M reviews). Reviews whose deck has an ancestor at distance k:

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| reviews reached | 49.21% | 38.29% | 31.20% | 20.93% | 7.80% | 1.62% |

Chain-depth mass sits at **depth 4 (13.13%)**, with depth 0 only 50.79%. So a single "parent deck"
link would leave most of the hierarchy unused -- Anki users nest decks (`Japanese::Core2k::Stage 1::Vocab`).
This is what set iter 50's L=3 rather than L=2: it is the smallest L that tests *tree* rather than
*parent*. Levels 4+ are held back because each costs a full extra 4-layer pass for a shrinking share
of the corpus (21 layer-steps at L=3 vs 13 today).

---

## ★★ THE CODE IS WRITTEN AND SMOKE-TESTED (2026-08-16) — what is left is Andrew's go, then GPU

`rwkv/id_features.py` + hooks in `rwkv/data_processing.py`, all behind **`RWKV_ID_FEATURES=1`,
default OFF and structurally inert when off** (with the flag unset the column list is the original
24, the width helper returns 92, and `parity3/parity_train_vs_rnn.py` still passes all seven cases).
The flag has to be inert rather than merely unused, because every live run reads the existing LMDBs
and must keep producing byte-identical input tensors.

**Shipped: 21 new columns replacing the card-state column → width 92 → 112.** Time-of-day raw phase
+ deviation from the user's running circular mean; true-phase day-of-week / day-of-year + weekend
flag; sub-day time-since-any-review; user tenure; creation→first-review; deck age, depth,
`card_predates_deck`, `is_default_deck`; the four creation-batch columns; `is_default_preset`.
All 21 are `keep_columns` — none describes the outcome, only *when* the review happens and what the
card/deck are, the same status `day_offset_diff` already has.

### ⚠ THREE CORRECTIONS TO THE PLAN ABOVE, EACH FORCED BY RUNNING IT

1. **★ "FIX (one line)" is wrong, and the documented version does not work.** Clamping
   `elapsed_seconds` to the **−1 sentinel** moves the NaN one column over instead of removing it:
   `elapsed_seconds_cumulative` is a per-card cumsum, so a card that gains a *second* sentinel
   cumulates to **−2** and `scale_elapsed_seconds_cumulative` takes the identical `log(negative)`
   branch. Measured on this page's own index case — user 486, card 1674953822938: raw
   `[-1, -1, 43164, 22550]` → cumulative `[-1, -2, 43162, 65712]`, NaN on row 9182. **The plan's fix
   would have passed a raw-column check and still lost the user.**
   **Clamp to 0 instead.** The page's stated objection ("that would claim the two reviews were
   simultaneous") is weaker than it looks: the overlap is bounded by the review's *own* duration, so
   the true gap really is ~0 seconds and −1 ("no previous review at all") is the *less* accurate
   code. It is also self-limiting — with every non-sentinel value ≥ 0 a card's cumsum is
   `−1 + sum(nonneg) ≥ −1` by construction, which `data_processing` now asserts rather than hopes.
   **Counterfactual over 60 stride-sampled train users: 4 of 60 (6.7%) would have NaN'd** — higher
   than the 3.2% recorded here.
2. **NOT `elapsed_days`.** The one-liner says "same for `elapsed_days`"; that would be a silent
   corruption, because `is_first_review` is defined as `elapsed_days == -1`, so clamping a mid-card
   review to the sentinel re-labels it as that card's FIRST review and poisons `cum_new_cards`,
   `diff_new_cards` and the label machinery. It gets an **assert** instead — integer days cannot go
   negative from a sub-day overlap, so if it ever fires we want to know loudly.
3. **★ The reference derivations in `optimization/feature_stats_id.py` LEAK, and porting them
   verbatim would have shipped it.** They count a user's **whole** card collection, which is correct
   for the marginal distributions they were written to measure and wrong as a feature:
   `creation_batch_1d` would tell the model how many cards the user *went on to* create later that
   day. The production counts are clipped at `review_time` — on user 1 that is **289 of 22,430 rows**
   that would otherwise have leaked. This is exactly the case the "Leakage rule" section above
   anticipated ("same-day-created-and-reviewed cards stay honest") and it was still one copy-paste
   away.

### Also fixed on the way: `card_features_dim = 92` was hardcoded TWICE
Once in `srs_model.py` and once in `srs_model_rnn.py` — two copies of one number, which is the shape
of bug this project keeps paying for (the Rust positional-stream bug; STRIP_CMIX living only in
`rwkv_model.py`). Both now call `id_features.input_width()`. And `RWKV_ZERO_FEATURES=22` is **refused**
under the new layout rather than silently masking whatever now sits at index 22 (`day_of_week`) —
the rebuild removes the state column at the source, so the mask is obsolete, not merely renumbered.

### The smokes (both CPU, both green)
* `scratchpad/id_features/smoke_id_features.py` — inertness when off; when on, finite values on
  60 users / 6.3 M rows (zero NaN, zero exceptions) plus three leakage properties: causal batch
  counts, the circular mean using strictly prior reviews, and **prefix invariance at exactly
  0.000e+00** (truncate a user's history, the surviving rows are unchanged — the strongest of the
  three, because any accidental whole-table statistic breaks it at once).
* `scratchpad/parity3/smoke_id_features_width.py` — training class, deploy RNN class and
  `CARD_FEATURE_COLUMNS` agree on the width under BOTH flag values (92 / 112). This is the §9
  three-way check for this flag; it cannot live in `parity_train_vs_rnn.py`, which is single-stack.

### WHAT IS STILL NOT DONE (in order)
1. **`scratchpad/parity3/smoke_scripted_eval.sh`** — the iter-48 guard, mandatory after touching
   `srs_model.py`. Needs a free GPU, so it waits for the running QAT job.
2. **The 100-user de-risk build** on `-id`, comparing new-columns-ON vs new-columns-OFF (the `-id`
   vs published comparison is invalid — `size` moves for ~30% of users from the dataset swap alone).
3. **`find_equalize_test_reviews` → a new `label_filter_db`**, and re-reading the `size` gate as
   *within a rebuild generation*.
4. **Andrew's go for the ~23 h train+test rebuild on F:.** Not started; the endgame order puts the
   algorithmic loop first, and every run in it reads `train_db_5k_h1`.
5. **Rust deploy debt:** `rust/rwkv-infer` has its own input width and will need the new columns
   plus a fresh parity trace. Not a blocker for measuring the features, but it is a gap the moment
   one is adopted.

