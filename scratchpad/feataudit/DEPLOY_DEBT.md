# The gen-5 feature vector has no deploy-side implementation

Found while auditing pre-processing, 2026-09-09. Companion to `FINDINGS.md`. This is a scope note
for phase 7, not a defect in anything that runs today.

## What is true right now

`review_features(&self, features: &[f32], state)` takes the input vector as a **slice**
(`optimization/DEPLOY_FUNCTIONS.md` section 1). The caller builds it. And nothing in this repo
builds it for deploy:

* **`rust/rwkv-infer` computes no feature at all.** Grepped for `tod_sin`, `sibling`,
  `creation_batch`, `tenure`, `deck_depth`, `cyc3_`, `dow_sin` across `src/` -- zero hits. The
  engine consumes a trace whose features are already computed.
* **`run_as_rnn.RNNProcess.get_tensor` reads `CARD_FEATURE_COLUMNS` off a row** that
  `data_processing.get_rwkv_data` produced (`run_as_rnn.py:144`). It does not derive them either.

**The good half of that, and it is the part §9 cares about: there is no SECOND implementation, so
there is nothing to diverge.** Under `RWKV_ID_FEATURES=1` the trace export takes the shared
`get_rwkv_data` path (`export_rnn_trace.py:89`), which is exactly why the gen-5 parity trace came
back self-contained at 0.000e+00. Train, eval and the deploy RNN all read one builder.

**The other half is unwritten work.** Against the published 92-dim layout, gen 5 adds **47 card
feature columns** (23 `NEW_COLUMNS` + 24 `CYCLE_COLUMNS`) and drops 2, and it removes the 28
pseudo-cycle encoding dims that `prepare_batch.add_encodings` used to synthesise. A live Anki
scheduler must produce all 47 from its own database. None of them exists outside
`rwkv/id_features.py`, which is pandas over a whole-user frame.

## What each new column needs at schedule time

| what it needs | columns |
|---|---|
| the wall clock only | `tod_sin/cos`, `dow_sin/cos`, `doy_sin/cos`, `is_weekend`, the 12 review-half cycle columns |
| the card's own id / first review | `scaled_creation_to_first_review`, `card_predates_first_review`, the 12 `_first` cycle columns |
| deck / preset metadata | `scaled_deck_age_at_review`, `card_predates_deck`, `is_default_deck`, `is_default_preset` |
| a deck-tree walk | `scaled_deck_depth` |
| the user's first review time (1 scalar) | `scaled_user_tenure` |
| the user's last review END time (1 scalar) | `scaled_t_since_any_review` |
| a running circular mean of prior review times (2 floats) | `tod_dev_sin/cos` |
| a sorted array of the user's card-creation times, binary-searched and clipped at now | `scaled_creation_batch_1min/1h/1d/pos_1h` |
| the last review END of a different card of the same note | `scaled_sibling_gap` |

## The thing that would have been a blocker, and is not

**None of this touches the frozen per-entity state budget** (9 B/card, 27 B/note). The extra
quantities are collection facts Anki already stores -- card ids are creation timestamps and the
revlog holds every review -- so they are QUERIES at schedule time, not model state that has to be
serialised beside the RNN state. `tod_dev` is the only one that wants running state, it is
per-USER, and the global stream is the one scope this project has always allowed to grow.

So the cost is implementation effort and per-review query time, not deploy bytes. Two items are
worth measuring before phase 7 commits to them, because they are the only ones that are not O(1):
the creation-batch binary searches (they need the user's card-creation times sorted once per
session, not per review) and the sibling lookup (one indexed revlog query per review).

⚠ And one correctness carry-over from `FINDINGS.md` W1: whatever implements `scaled_sibling_gap`
must reproduce the **sentinel** for "no preceding sibling review" exactly -- it is written as
standardized `0.0`, not as a missing value, on 97.4% of rows. A deploy implementation that emits a
true zero gap, or a NaN, for that case is silently wrong on almost every review.
