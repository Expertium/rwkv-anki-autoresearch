# Future input features from real timestamps (planning only — needs a new dataset export)

**Context (Andrew + Claude, 2026-07-15).** anki-revlogs-10k is anonymized: `day_offset`
integer days only, re-indexed IDs. In real Anki, `card_id` / `note_id` / `deck_id` /
`review_id` are all **epoch-millisecond creation/review timestamps**, so a fresh export that
preserves them unlocks the features below. This breaks the current 92-dim input invariant →
a **future data-side phase**, not something to A/B on the current LMDBs. Deploy-side cost is
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
| med | note_id/deck_id/preset_id ages: card − deck creation, card - preset creation, deck age at review, preset age at review | Early core card vs late addition; preset ids are creation timestamps too (Andrew 2026-07-16: use both). ⚠ the DEFAULT deck and DEFAULT preset both have id 1 (constant, not a timestamp) — derive an is-default flag for those instead of an age. Andrew 👍 |
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
(inherited from upstream). Cost to use it = an LMDB rebuild, not a data rebuild.

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

## Leakage rule
All count/batch features must be computed **as of review time** during preprocessing (not from
the full table) so same-day-created-and-reviewed cards stay honest.
