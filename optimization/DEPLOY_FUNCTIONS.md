# The two functions CPU optimization is measured on

Andrew, 2026-07-26: the deploy-speed work is anchored on **named functions from the Anki fork** —
one that computes the five per-entity states, one that computes p(recall). This file records
exactly which they are.

**Only `review_features` is actually SCORED** — see [`DEPLOY_SPEED_LOG.md`](DEPLOY_SPEED_LOG.md),
which holds the single table. Andrew's call once it became clear the two overlap (§⚠ point 1
below): `retrievability_head` is a sub-computation of `review_features`, so a second table would
measure part of the same work twice. It stays documented here for reference.

Source: `vendor/jschoreels_anki/rust/mod.rs` — the READ-ONLY reference copy of
[github.com/JSchoreels/anki](https://github.com/JSchoreels/anki), an Anki fork that ships the old
2.76M RWKV as a live scheduler. Verified against the file, not from memory.

## 1. States — `review_features`

```rust
fn review_features(&self, features: &[f32], state: SrsStateRef<'_>) -> ReviewHeads
```
`vendor/jschoreels_anki/rust/mod.rs:2944`

One call consumes one review and produces **all five stream states**, chained in arch order —
`card → deck → note → preset → global`:

```rust
let x = self.feature_mlp(features);
let (x, card_state)   = self.modules[0].run(&x, state.card);
let (x, deck_state)   = self.modules[1].run(&x, state.deck);
let (x, note_state)   = self.modules[2].run(&x, state.note);
let (x, preset_state) = self.modules[3].run(&x, state.preset);
let (x, global_state) = self.modules[4].run(&x, state.global);
```

Note the order is card→**deck**→**note**→preset→global, matching our `architecture.py` module
order (`card_id, deck_id, note_id, preset_id, user_id`) — *not* the `RWKV_SUBMODULES` order. This
is the same trap CLAUDE.md §"ERRATUM" records; the fork gets it right, and so must any comparison
against it.

**Unit: states/s.** One call = 5 stream states. Count *calls* per second and label it states/s
(the convention already used in `cpu_speed_log.md` §"What the throughput numbers MEAN").

**Our counterpart:** `Model::review` (candle, B=1) and `FastModel::review_batched` (the default
fast path) in `rust/rwkv-infer/src/{model,fast}.rs`. Both return the five new stream states plus
the heads, so they are the like-for-like target.

## 2. p(recall) — `retrievability_head`

```rust
fn retrievability_head(&self, prehead_x: &[f32]) -> f32 {
    1.0 - self.button_probabilities_head(prehead_x)[0]
}
```
`vendor/jschoreels_anki/rust/mod.rs:2940`

p(recall) = **1 − P(Again)**, off the rating head. `button_probabilities_head` (`mod.rs:2930`) is
`head_p_0 → relu → p_linear → softmax`, and index 0 is Again.

**Unit: probabilities/s.** One call = one p(recall).

**Our counterpart:** `Model::imm_prob` / `FastModel::imm_prob` — `1.0 - softmax(out_p_logits)[0]`,
the identical quantity. This is our benchmark's `imm` mode.

## ⚠ Three things to keep straight when measuring these

1. **They overlap — do not add their costs.** `review_features` computes retrievability *inline*
   (`let retrievability = 1.0 - button_probabilities[0];`), so `retrievability_head` is a
   sub-computation of it, not a disjoint stage. Timing them separately is still meaningful (they
   answer different product questions: building state vs re-scoring a due queue at fixed state),
   but their rates cannot be summed into a per-review total.
2. **The two rates differ by orders of magnitude, by design.** `review_features` runs the whole
   5-stream recurrence; `retrievability_head` is two small matmuls plus a softmax on an existing
   prehead vector. A shared y-axis or a single "speedup" figure across both would mislead.
3. **`prehead_x` is an input, not state.** `retrievability_head` can be benchmarked in isolation
   from a fixed prehead vector — which makes it the cleaner target of the two, since its timing is
   independent of state chaining, entity routing and the stored curve.

## ⚠ Licensing

`vendor/jschoreels_anki/` is **AGPL-3.0-or-later** (see its `NOTICE.md`). *Naming* these functions
and benchmarking against their structure carries no licensing consequence. **Copying any of their
implementation into `rust/rwkv-infer` would make our engine AGPL-derived** — fine for shipping
inside Anki, but it must be flagged to Andrew first and carry provenance comments. That applies in
particular to the AVX2/FMA `x86_simd.patch`, which is the most tempting thing in there.

---

## ⚠⚠ FILTERED (CRAM) DECKS: A DEPLOY REQUIREMENT NO GATE CAN CATCH (Andrew, 2026-08-26)

Andrew's question after we measured that filtered decks are absent from the training data:
*"since filtered decks aren't in the training data how do we make sure they are handled
correctly during deployment?"* They are absent from the data only because Anki's export
resolves them away; they are very much present at deploy time.

**WHAT TRAINING ACTUALLY SAW** -- three facts, checked in code, that together make the
requirement precise:

1. **Filtered REVIEWS are in the training data.** `filter_revlog` drops only
   `review_kind == 3 AND ease_factor == 0` (filtered-deck *rescheduling* artifacts). A real
   review answered inside a filtered deck is kept
   (`scratchpad/dataset_id/build_parquet_id.py:58-59`).
2. **They are attributed to the HOME deck.** Anki's export resolves `odid`, verified by
   measurement rather than by reading Anki: cards referencing a deck absent from the decks
   table are 7 of 1,608,074 (0.0004%), and no deck carries the `preset_id == 0` filtered marker
   (`scratchpad/hybrid100k/filtered_deck_probe.py`).
3. **Their filtered-ness is INVISIBLE to the model.** `review_kind` becomes `scaled_state`,
   feature column 22, and the champion sets `RWKV_ZERO_FEATURES=22`. The rebuild drops the
   column outright.

=> **Training's view is "a cram review is an ordinary review, in the home deck."** The deploy
requirement is therefore to reproduce exactly that, not to add special handling.

**THE REQUIREMENT, for whoever writes the Anki integration:**

* Resolve the home deck before filling `ReviewInput.deck_id`:
  `if odid != 0 { odid } else { did }`.
* Resolve the **preset the same way** -- the home deck's preset, not the filtered deck's config.
  A filtered deck schedules from its own config, which is exactly the value NOT to pass.
* Do **not** create a deck entity for the filtered deck, and do **not** flag cram reviews.
  Training cannot see the flag, so supplying one is a divergence, not an improvement.

**WHY NOTHING WILL CATCH THIS.** `review_features` takes `deck_id` as an INPUT
(`ReviewInput.deck_id`, mod.rs:357); the engine never looks it up. Pass the filtered deck's id
and every call still succeeds. The failure is silent and expensive: each cram session invents a
brand-new deck entity whose stream state starts at zero, discarding the accumulated home-deck
state, and Anki then deletes the filtered deck so that state is thrown away. Predictions are
simply worse during and after cramming, with no error anywhere.

**The reference fork demonstrates the correct resolution but only in its SCAN path** --
`scan_bench_current_deck_id_sql` (mod.rs:6715-6721) emits
`case when c.odid != 0 then c.odid else c.did end`, with a fallback to bare `c.did` on schemas
lacking the column. Its LIVE path delegates to whoever constructs `ReviewInput`; every in-file
construction site is a test. So this is a pattern to copy, not a guarantee we inherit.
⚠ Note also mod.rs:6696 passes `preset_id: Some(row.deck_id)` in that bench path -- a deck id
used as a preset id. Their code, their bench, but do not copy that line.

**HONEST RESIDUAL, not a defect:** the model cannot distinguish deliberate cramming from
studying ahead, because the only column that marks it is zeroed. It infers what it can from the
short elapsed time, which IS an input. That limitation is shared by training and deploy, which
is what makes deployment *consistent*; whether cram deserves distinct treatment is a research
question and would need the state column back.
