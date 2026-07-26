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
