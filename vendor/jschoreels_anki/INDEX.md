# Function map — what JSchoreels/anki has, and what we want from it

Andrew asked specifically for **the functions that compute card/note/deck/preset/global
state and p(recall)**. All line numbers refer to the fetched copies in `rust/` and
`python/` (branch `main`, 2026-07-25). License caveat: see `NOTICE.md` (AGPL-3.0+).

## The architecture it implements = the OLD upstream model, hardwired

`rust/mod.rs:19-29`:
```
D_MODEL = 128, CARD_FEATURES = 92, HEADS = 4, HEAD_SIZE = 32, HEAD_DIM = 4*D_MODEL
NUM_CURVES = 128
MODULE_LAYERS      = [3, 4, 2, 3, 4]      // card, deck, note, preset, global
CHANNEL_MIXER_DIMS = [192, 256, 192, 256, 256]
```
So it is the 2.76M-param upstream RWKV with the ORIGINAL curve head — **not** our track-2
lineage (our A15 champion is d=96/H=3, `MODULE_LAYERS = [2,4,1,3,3]`, channel-mixer factor
1.0, GRU curve head, 9 stripped mixers). Everything is `const`, so his engine cannot load
our checkpoints without parameterization — the same gap our own `rust/rwkv-infer` has.

## 1. The five-stream state chain — `rust/mod.rs`

| what | where | notes |
|---|---|---|
| **`SrsModel::review_features(features, state) -> ReviewHeads`** | `mod.rs:2944` | **THE function Andrew asked about.** `feature_mlp(features)` → `modules[0].run` (card) → `[1]` deck → `[2]` note → `[3]` preset → `[4]` global, each returning `(x, next_state)`; then `prehead_norm`, `curve_head`, `button_probabilities_head`. Same chain order as ours. |
| `SrsStateRef` / `SrsStateOwned` / `SrsState` | `mod.rs:3084-3131` | the 5 slots as plain fields (`card, deck, note, preset, global`), plus `module(id)` mapping 0..4 |
| `ReviewStateMaps` | `mod.rs:3135` | `HashMap<i64, ModuleState>` per entity kind + one `Option<ModuleState>` global — the per-user store Anki persists |
| `ModuleState::run(input, state) -> (Vec<f32>, ModuleState)` | `mod.rs:3677` | one stream: loops its layers carrying `x` **and `v0`** (the v0-mix residual), returns the new stream state |
| `RwkvLayer::run` / `TimeMixer::mix_parts` / `mix_output` | `mod.rs:3800+, 4071, 4157` | the per-layer RWKV-7 recurrence (r/k/v/a/g, decay, bonus) |
| `LayerState { time: Option<TimeState>, channel_shift: Option<Vec<f32>> }` | `mod.rs:3886` | exactly our state decomposition: WKV matrix + the two token shifts |
| `state_for_card` / `restore_state` / `cache_state` / `restore_cache_state` | `mod.rs:1129-1166` | persistence API (serialize per-entity states to bytes) |
| `serialize_module_state` / `deserialize_module_state` | `mod.rs:3302, 3332` | the on-disk state format |
| `StateCompression { shift_bits, matrix_rank, matrix_bits, power_iterations }` | `mod.rs:5873` | **his state-compression scheme — low-rank + bit-quantized WKV matrix + quantized shifts, i.e. the same design as our q72u** (worth diffing against ours) |

## 2. p(recall) and the heads — `rust/mod.rs`

| what | where | notes |
|---|---|---|
| `button_probabilities_head(prehead_x) -> [f32; 4]` | `mod.rs:2930` | `head_p_0` → ReLU → `p_linear` → softmax = the 4-way Again/Hard/Good/Easy head |
| **`retrievability_head(prehead_x) -> f32`** | `mod.rs:2940` | **p(recall) = `1.0 - button_probabilities[0]`** — identical definition to our `out_p_binary` |
| `curve_head(prehead_x) -> ReviewCurve` | `mod.rs:2911` | `head_w_0` → ReLU → `head_w_norm` → `head_w_4` → `w_linear` → softmax = the 128-basis mixture weights; plus `head_ahead_0` → ReLU → `ahead_linear` = the piecewise-linear ahead residual **we disabled** (RWKV_NO_AHEAD_RESIDUAL) |
| `predict_retrievability_many*` | `mod.rs:669-759` | batched p(recall) at many horizons: plain, from warm-up state, after a review, after several reviews |
| `review` / `predict_many` / `review_many` | `mod.rs:576, 624, 2777` | single and batched review entry points |
| `warm_up_reviews` (+ `_sequential`, `_with_state_compression`) | `mod.rs:883-1016` | replay a card's history to rebuild state — the cold-start path Anki needs |

## 3. What's directly useful for OUR CPU-speed work (roadmap step 4)

1. **`rust/x86_simd.patch`** (branch `codex/rwkv-x86-simd`, +214 lines) — AVX2+FMA versions
   of the two hot primitives, `dot_product` and `add_scaled_in_place`, with
   `is_x86_feature_detected!` runtime dispatch, scalar fallback, and tests asserting
   bit-comparable results. **Our `rust/rwkv-infer` has NO SIMD at all** (checked: no
   `avx2`/`fma`/`target_feature` anywhere in its 3,533 lines), so this is the single most
   portable win available to us, and Andrew's RTX-4070 box is x86_64 with AVX2.
2. **Batched query path** — the macOS build routes through `matmul.rs` (Accelerate
   `cblas_sgemm`) with `LayerQueryBatchScratch` / `review_many` / `bulk.rs`, i.e. it turns
   many per-review matrix-vector products into GEMMs. That is exactly the fix for the
   overhead-bound behaviour measured in `optimization/CPU_INFERENCE.md`. On x86 he falls
   back to the scalar path — so a batched x86 GEMM (via `matrixmultiply`/`gemm` crate or
   hand-rolled) is an unclaimed win for us.
3. **`rwkv_bench.rs` / `rwkv_predict_bench.rs`** — ready-made CPU benchmark harnesses to
   copy the methodology from (ours is `optimization/measure_throughput.py` + the 3-user
   reference set).
4. **`StateCompression`** — an independent implementation of low-rank + quantized state to
   sanity-check our q72u numbers against.
5. **`python/srs_model_rnn.py` (`review()` at :116, `forgetting_curve()` :75, `interp()`
   :94)** — a trimmed deploy-shaped copy of our own `rwkv/model/srs_model_rnn.py`; useful as
   a second opinion on what can be dropped from the inference path.

## 4. Gaps to close before any of it runs our champion

Both his engine and ours hardwire the old shape. To measure whether our ablations bought CPU
speed we need, in `rust/rwkv-infer`: parameterized `d_model`/`HEADS`/`MODULE_LAYERS`, the
**GRU curve head** (`RWKV_GRU_HEAD`), `RWKV_STRIP_CMIX` (dummy mixers), `RWKV_STRIP_L0_VLORA`,
and the no-ahead-residual curve. None of that exists in his fork either — it is our work,
but his file layout is a good template for where each piece goes.
