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
