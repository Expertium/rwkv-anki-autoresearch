# Datasets — what exists, what's in it, what it's good for

Single place for "which review dataset should we train on". Written 2026-07-27 after
Andrew asked whether the old 20k set was worth training on. Supersedes the scattered
DATA FACT bullets in CLAUDE.md for questions of *dataset choice* (those bullets stay
authoritative for the `-id` build details).

## The four sets on this machine

| Set | Where | Users | Reviews | note/deck/preset | Real timestamps |
|---|---|---|---|---|---|
| `anki-revlogs-10k` | `C:\Users\Andrew\` | 10,000 | ~745 M | **yes** | no (anonymized day counter) |
| `anki-revlogs-10k-id` | `C:\Users\Andrew\` | 10,000 | ~728 M | **yes** | **yes** (epoch-ms) |
| `anki-revlogs-3k` | `C:\Users\Andrew\` | 3,000 | — | yes | no |
| **FSRS-Anki-20k** | `F:\FSRS\FSRS-Anki-20k` | **20,000** | **~1.65 B raw** | **NO** | yes (epoch-ms) |

Everything trained so far uses `anki-revlogs-10k`. `-id` is the drop-in source for the
timestamp features (same 1:1 user numbering) — see `FUTURE_FEATURES.md`.

## FSRS-Anki-20k — VERDICT: do not train on it (2026-07-27)

**What it actually contains** (richer than remembered — the raw side, not the stripped
`dataset/` CSVs):

- `revlogs/{1,2}/<user>.revlog` = protobuf `RevlogEntries`, 57.4 GB, 20,000 users.
  Per entry: `id` (**real epoch-ms answer time**), `cid` (**real epoch-ms card id**),
  `button_chosen`, `interval`, `last_interval`, `ease_factor`, `taken_millis`,
  `review_kind`; per file `next_day_at` (the user's day-rollover, i.e. local cutoff).
- `dataset/` = the stripped 4-column form (`card_id, review_th, delta_t, rating`) —
  ignore it, the raw side is strictly better.
- **Missing: `note_id`, `deck_id`, `preset_id`.** That is the whole gap.
- ~1.65 B raw entries estimated (57.4 GB / 34.7 bytes per entry, measured on a
  40-user sample) = ~2.2x the 10k set.

**It would ingest without any code change.** `data_processing.py:210-225` already
handles missing ids — missing note -> unique per-card placeholder, missing
deck/preset -> one shared placeholder, and the three `*_is_nan` flags are model
inputs (features #9). The hierarchy invariant survives.

**Four reasons not to, in order of decisiveness:**

1. **DISK — it does not fit.** 372.5 GB of LMDB per 5,000 users, so 20,000 users is
   **~1.5 TB**. Free space is C: 243 GB + F: 890 GB = 1.13 TB, and 605 GB of F: is
   already earmarked for the new-features rebuild. Preprocessing is 2-4 days of CPU
   for 5k train+test, so 20k is over a week — and would need redoing after the
   feature change.
2. **It trains a regime we never serve.** In the 10k set only **66 / 10,000** users
   have no cards file at all and **1.1%** of cards have a deck missing from the decks
   table (measured over 1,363,468 cards in a 60-user sample). So "no note/deck/preset"
   is ~1% of our training data today and **0% of deployment** — a real Anki collection
   always has them. Adding the 20k makes ~2/3 of the gradient come from that regime,
   with three of five streams degenerate (note -> per-card, deck and preset ->
   per-user). At 558k params with capacity now binding (the A18 LoRA-halving sign
   flip), that is capacity spent in the wrong place.
3. **Leakage.** ~**4.3%** of the 20k users are already in the 10k set (869 / 20,000,
   a LOWER bound — see the method note below), of which **244 are in the eval half
   5001-10000** and **132 in the VAL subset 5001-7500**. Fixable by exclusion, but it
   is one more thing to get right. (Cross-check: two independent samples of 10k and
   20k from a ~230k-collection population would overlap by ~1,000 users, so 869 is
   about what random sampling predicts — the true figure is probably close to it.)
4. If it were ever wanted anyway, the only sane recipe is **pretrain on 20k ->
   finetune on 10k** (big-noisy then small-clean), never a blended mix.

### ⚠ Method note — `card_id` is USELESS as a collection fingerprint

Anki card ids are epoch-ms of card creation *in the collection that made the card*,
and **shared decks carry them to every downloader**. Measured here: one card id is the
first card of **636 different** 20k users, and a naive card-id match reported 64%
overlap (12,795 / 20,000) mapping onto only 1,157 distinct 10k users — a many-to-one
tell that the "matches" were shared decks, not shared collections.

**Use the review timestamp instead.** A review is performed locally, so its epoch-ms
is unique to the collection. On the 20k side that is the raw `id`; on the `-id` side
the stored `review_time` is `id - taken_millis`, so the comparable value is
**`review_time + duration`**. Tools: `scratchpad/ds20k/scan20k.py` (hand-decodes the
protobuf header, 64 bytes per user instead of ~2.7 MB) + `overlap2.py`.
`overlap.py` is the card-id version, kept only as the counter-example.

The 4.3% is a lower bound because it tests only whether the 20k user's **first raw
review** survived the 10k build's filters (manual / filtered-deck rows and non-latest
learning sequences are dropped).

## ⚠ "More epochs" is not evidenced either — augmentation is OFF, so epochs are byte-identical replays

`prepare()` calls `torch.manual_seed(seed)` at the top of every batch
(`prepare_batch.py:210-211`) and `prepare_data_train_test` passes the **same constant**
`fixed_seed` for every batch (`:655`), which under `RWKV_AUGMENT_SEED=1234` is 1234.
The two augmentations — per-batch random ID codes and the per-batch random cycle-phase
baseline (`INPUT_FEATURES.md` notes) — are therefore drawn **identically in every
epoch**. Epoch 2 replays the same tensors as epoch 1; only dropout differs.

**Consequence for the record:** the `champ5k_b1` budget A/B ("the 2nd epoch adds
nothing" — ahead -0.00006 p=0.31, imm +0.00043 better p=6e-62), which is why WS is
fixed at 1 epoch, was measured in exactly the configuration where extra epochs *cannot*
add anything. It establishes "more **identical** epochs don't help", not "more epochs
don't help", and must not be quoted as the latter. This is the same point CLAUDE.md's
endgame item (b) raises for the 10x run, now confirmed in code rather than suspected.

**★ UPDATE 2026-08-16 — the fix is NOT "turn augmentation on", and the premise survives anyway.**
Andrew decided augmentation stays OFF ("screw augmentation, at least that particular kind"), so
epochs remain byte-identical replays by choice. That does not leave "more epochs" unevidenced,
because the **2026-08-11 budget calibration measured it directly, with augmentation off**: a
3x-budget step is worth **+0.002**, projecting to **+0.0042 at 10x** against the +0.0040 upstream
gap. So what extra epochs buy in this setup is optimization steps under the WSD schedule (plus
per-epoch dropout variation), not data variety — and the `champ5k_b1` null is a statement about
2 epochs at that budget, not about 12.5.
⚠ Two further reasons augmentation-on was a bad trade regardless: ~0.0024 run-to-run variance
against a 0.0001 accept gate, and structural incompatibility with KD-from-dump (the dump's
`labels_sum` checksum proves LABEL alignment and is blind to input-side changes).

## The cheap experiment that settles the data-vs-epochs question

Neither "more users" nor "more epochs" is currently supported by evidence. Hold total
gradient steps fixed and trade users for epochs:

- **reference:** 5,000 users x 1 epoch = 22,346 WS steps — this is iter 31, already run.
- **candidate:** 2,500 users x 2 epochs ~= the same step count, same decay ratio, same
  VAL half (5001-7500).

`get_groups(db_path, db_size, max_train_global_len, users)` takes a user list and
`TRAIN_USERS_START/END` come straight from the toml, so this is a **two-line config
change** — no LMDB rebuild, no disk, ~6-7 h.

- Halving the users costs a lot -> **data-limited** -> the 20k's extra users are worth
  solving the disk problem for (via pretrain->finetune).
- Halving costs ~nothing -> data is not the bottleneck at 5,000 users -> neither the
  20k nor the 10x-epoch run is the right lever. Worth knowing **before** committing
  ~4 days of GPU to the latter.

## Free high-quality data that already exists

`train_db_5k_h2` (users 5001-10000, 372.5 GB, **already built** on F:,
`data_processing_train_5k_h2.toml`) doubles the fully-featured data at zero
preprocessing cost — but it **is** the eval set, so it is usable only for the final
shipped model after research closes, exactly as upstream does with its two checkpoints
(trained on 101-4999 and 5000-10000).

## The interval definition: `elapsed_seconds` is answer-to-answer, and it should be end-to-start

Andrew, 2026-08-29: *"given that the dataset has different interval lengths now and it matters for
same-day reviews, once we're done with RWKV, should we make a third table for srs-benchmark?"*

### The three definitions

A revlog row is written when the user **answers**, so `id` is the answer time and `duration` is how
long that review took. Writing `show(k) = id(k) - duration(k)`:

| definition | formula | who uses it |
|---|---|---|
| answer-to-answer | `id(k) - id(k-1)` | **the public dataset's `elapsed_seconds`** (`build_parquet.py`: `review_time = entry.id`, then `.diff()`) |
| show-to-show | `show(k) - show(k-1)` | the `-id` set's naive diff |
| **end-to-start** | `show(k) - id(k-1)` | what memory decay actually spans |

    end_to_start = elapsed_seconds - duration(k)

⚠ It is the **current** review's duration that comes off, not the previous one. Show-to-show is the
one that differs by `duration(k-1)`. Easy to get backwards.

### ★ It needs NO new dataset

`duration` and `elapsed_seconds` are BOTH columns of the public `anki-revlogs-10k`. The corrected
interval is a one-line transform of data everyone already has -- no `-id` build, no HF upload, no
reprocessing. That is what makes this proposable upstream at all.

### Measured, 40 stride-sampled users / 2.18 M reviews (`scratchpad/hybrid100k/interval_def_effect.py`)

| | same-day (30.8% of rows) | longer interval (69.2%) |
|---|---|---|
| median gap | 485 s | 531,311 s |
| duration as a fraction of the gap, median | **1.70%** | 0.0012% |
| ...p90 | **13.83%** | 0.01% |
| ...p99 | **65.22%** | 0.07% |
| rows shrinking >= 10% | 13.9% | 0.0% |
| corrected gap goes NEGATIVE | **0.559%** | 0.000% |

**The effect lives entirely in a TAIL, and a median-only read would dismiss it.** 1.7% sounds
negligible; 13.9% of same-day rows moving by a tenth or more does not, and those are the short
learning-step reviews where short-term modelling is actually differentiated. (Same shape as the
median-vs-max error that cost iter 51.)

**On longer intervals the correction is numerically invisible**, so it cannot touch the
"Without same-day reviews" table at all.

### ★ THE REAL ARGUMENT IS NOT PRECISION, IT IS THAT `elapsed_seconds` IS NOT KNOWABLE AT PREDICTION TIME

`elapsed_seconds(k) = end_to_start(k) + duration(k)`. At the moment of prediction the user has been
shown the card and has **not answered yet**, so `duration(k)` does not exist. The benchmark feeds it
anyway, inside the interval -- and `duration(k)` correlates with the outcome, because a review the
user struggles with takes longer. So the current interval carries a whisper of the label.

This is the same issue our own DEPLOY CONTRACT already handles by **zeroing the most recent
review's duration** (Andrew, 2026-07-27). We removed `duration(k)` from the duration column and left
it inside the interval column.

That reframes the question. It is not "is a more precise interval nicer", it is **"is a
prediction-time-unavailable, outcome-correlated quantity being fed to every algorithm"**. The answer
is yes, for 30.8% of rows, by a median 1.7% and a p90 of 13.8%.

### RECOMMENDATION: not a third table -- a measurement, then probably a FIX to the second one

1. **A third table would be near-duplicate.** It cannot differ from "Without same-day reviews" at
   all, so it could only ever shadow "With same-day reviews". Two nearly-identical tables invite the
   reader to pick one, which is the wrong framing for what is a correction rather than a variant.
2. **Measure before proposing.** Re-run 2-3 algorithms (FSRS-6, FSRS-7, and one seconds-aware
   baseline) with `elapsed_seconds - duration` and see whether any RANKING moves. Cheap: one-line
   transform, no new data.
3. **PRE-REGISTERED PREDICTION: aggregate LogLoss moves less than the gap between adjacent rows of
   the with-same-day table, and no ranking changes.** A 1.7% median change in `t` moves a forgetting
   curve very little. If that is right, this is a footnote and a `--interval-def` option, not a
   table. If a ranking DOES move, it is a correction worth making properly.
4. **The negative rows need a stated rule first.** 0.559% of same-day rows have `duration(k)`
   exceeding the whole recorded gap. Clamp to 0 (the same choice `FUTURE_FEATURES.md` reached for
   the NaN landmine, and for the same reason: the real gap is bounded by the review's own duration,
   so it genuinely is ~0). Do not leave it implicit.
5. **Disclose the interest.** RWKV tops both tables today and is one of the seconds-aware
   algorithms, so a short-interval correction is not a neutral proposal coming from us. Say so in
   the issue, and let the measurement stand on its own.

**Timing: after the RWKV work, as Andrew said.** Nothing here is urgent, and the measurement is
worth more once we can also report what it does to our own entry.

### DECIDED (Andrew, 2026-08-29): a FLAG, not a third table

> *"Yeah, having a flag for it seems reasonable."*

So the upstream proposal is an option alongside the existing `--secs`, not a new results table.
Scheduled AFTER the RWKV work; nothing here is started.

**The hook is ONE site, three lines.** `features/base.py::_process_time_intervals` is the only
place `delta_t_secs` is derived (everything else matching `elapsed_seconds` in the tree is inside
`.venv`):

    if self.config.use_secs_intervals:
        df["delta_t_secs"] = df["elapsed_seconds"] / 86400
        df["delta_t_secs"] = df["delta_t_secs"].map(lambda x: max(0, x))

Two facts that make this cheaper than expected:
* It already sits behind `use_secs_intervals` (the `--secs` flag), so the new option goes next to
  an existing one rather than introducing a new axis.
* **It already clamps at 0.** So the negative-row rule I recommended for the 0.559% of same-day
  rows where `duration(k)` exceeds the whole gap is the behaviour the code ALREADY has -- the
  subtraction inherits it for free, and nothing new has to be specified.

`delta_t` (days) stays untouched, which is correct: the correction is invisible at day resolution
(median 0.0012%).

**Order of work when it starts:** implement the flag, run the 2-3 algorithm comparison, and only
then open the issue -- with the measurement, the pre-registered prediction ("no ranking changes"),
and the disclosure that RWKV is ours and is seconds-aware.

