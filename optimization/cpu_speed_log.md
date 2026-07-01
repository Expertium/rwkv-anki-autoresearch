# CPU inference-speed optimization log

Append-only log for **CPU throughput** optimization of the Rust candle inference engine
(`rust/rwkv-infer`) — the engine that ships in Anki AND produces card/note states for the
state-quant loop. Goal: maximize reviews/s, **single-thread first**, then multithreading.

## Protocol (Andrew, 2026-06-30)

- **Metric = the DEPLOYMENT workload: batched query throughput at B=128, 1 thread.** Anki scores
  the due-card queue with the read-only batched forward (`--bench-synth <secs> 128`), NOT the B=1
  sequential replay. The throughput-vs-RAM Pareto **knee is B=128** (see frontier below). Baseline
  champ_h2k16 ≈ **15,970 rev/s** @ 16 MB, single thread. (The B=1 full sequential replay `--bench`
  ≈ 408 rev/s is the *export/sibling state-production* workload, tracked separately when relevant.)
- **Paired simultaneous trial (the unit of the test):** launch the **champion** binary and the
  **candidate** binary *at the same instant*, each pinned to **1 thread** (`OMP=RAYON=1`), each
  running the bench for `<secs>` on the same weights (`champ_h2k16`). Each prints its review count
  → one paired point `(champ, cand)`. Simultaneity makes any outside load (FSRS benchmark,
  scheduler) hit both sides equally, so the *sign* of the difference is clean.
- **Why per-trial (not per-batch / per-user):** the signed-rank test needs independent pairs under
  matched conditions. Per-trial gives N=20 independent pairs with huge power (20 same-sign pairs →
  p≈2⁻²⁰); per-user would give only N=3 (ref traces) and conflate user-specific throughput; the
  batch is FIXED at 128 (the trial unit is the timed run, not a batch).
- **Test:** one-sided Wilcoxon signed-rank (H₁: candidate faster), **20 trials** (drop warm-ups).
- **Accept iff** `p < 0.01` **AND** median speedup > 1 **AND** correctness holds (candidate's
  `rust_pred_107.json` matches the champion's within 1e-5; pure-perf changes should be bit-identical).
- **Driver:** `python optimization/wilcoxon_speed.py --threads 1 --trials 20 --bench-mode synth
  --batch 128 --secs 4 --before champ_h2k16 --after champ_h2k16 --before-bin <champion.exe>
  --after-bin <candidate.exe>`. On ACCEPT: promote `candidate.exe` → the champion snapshot
  (`scratchpad/cpu_bench/champion.exe`). On REJECT: revert the source change; snapshot stays.
- Champion snapshot binary lives at `scratchpad/cpu_bench/champion.exe`; candidate is the freshly
  built `rust/rwkv-infer/target/release/rwkv-infer.exe`.

**Throughput-vs-RAM Pareto frontier (champ_h2k16, 1 thread, redone 2026-06-30** after H=2/K=16 halved
the card state — `scratchpad/cpu_bench/pareto_h2k16_champ.csv` + `pareto_h2k16_vs_old.png`):

| B | rev/s | RAM MB | rev/s/MB | note |
|---|---|---|---|---|
| 64 | 13,983 | 11.3 | 1243 | best rev/s per MB |
| **128** | **15,970** | **16.0** | 999 | **knee — deployment point** |
| 256 | 16,924 | 25.4 | 667 | raw peak (poor RAM trade: +59% RAM for +6% rev/s) |
| 2048 | 11,418 | 155.4 | 73 | cache-spill region |

vs the old iter36 model (H=1/K=32): RAM per B ~**halved** (B=128: 29→16 MB), throughput ~**1.5×** up,
peak moved B=128→256. The new frontier strictly dominates the old. **Engine is dim-agnostic** (derives
H,K,C,layers from weight shapes) → runs champ_h2k16 with NO Rust port.

**Profile @ B=128, 1 thread** (RWKV_PROFILE, since-reverted), per `review_batched` call ≈ 7560µs:
features2card **3.2%** · 5 streams (14 layers) **93.7%** · heads **3.1%**. Streams are uniform at
**~500µs/layer** (deck 4L=2006µs, note/preset/user 3L≈1500µs, card 1L=559µs) → ~50 candle ops/layer on
(128,32) tensors ≈ **~10µs/op of dispatch+alloc overhead** (not FLOPs — the compute is <1µs). So the
lever is **cutting per-layer op count + allocation** in time_mixer/channel_mixer: fuse the norms (each
layer_norm = 9 ops, ~42 norm-calls/review), cache the `format!`+HashMap weight lookups, reuse buffers.
Heads (out_w curve + out_ahead) are kept — deployment needs them for scheduling, and they're only 3%.

## Feature export (`export_features_fast.py`) — pandas anti-pattern, NOT a Rust job

**Profiled 2026-06-30** (cProfile, user 150, 6564 reviews): total 324s, of which **`Series.__setitem__`
242s (75%)**, dominated by `_setitem_with_indexer_missing` 199s — i.e. **assigning NEW keys to a pandas
Series** (the per-row `add_same` feature builder), which **rebuilds the arrow-string index O(n) every
assignment**. The torch ops (`get_tensor`) were only 26s (8%). => the bottleneck is a pandas misuse, NOT
compute -> **a Rust rewrite is NOT worth it**; the random id-encoding (torch-seeded) would also make a Rust
port's traces hard to keep bit-identical.

**Fixes (all bit-identical -- verified `np.array_equal` vs reference trace, no feature change):**
1. **dict not Series** (`rwkv/run_as_rnn.py` `add_same`): `row = dict(row)` -> O(1) inserts (was the 199s
   index-rebuild); `get_tensor` uses `[row[c] for c in COLS]`; dropped 2 redundant `Series.copy()`/review.
2. **vectorized day-offset** (`add_day_offset_encoding`): one `torch.sin`/`cos` over all 7 periods instead
   of ~50 tiny per-period 1-element torch ops; precomputed `_do_periods`/`_do_f`; float32-promotion-matched
   so bit-exact. (24.5s -> 17.2s.)
3. **★ direct partition read** (`export_features_fast.py`): `cards`/`decks` were read via
   `read_parquet(dir, filters=[("user_id","=",u)])` which **re-discovers all ~10k partition dirs every
   call** (the 8.4s `_filesystem_dataset`). Read the direct partition path `DATA/cards/user_id=<u>` like
   `revlogs` already does -> no discovery. (17.2s -> **8.3s**.)

**Per-user: >120s (timed out) -> 8.3s = ~15x**, bit-identical. Both the export and the inference path
benefit. **Rust rewrite NOT needed** (it was never compute -- it was a pandas anti-pattern + pyarrow
partition re-discovery). Multiprocessing across users (`scratchpad/export_mp.py`, round-robin, resumable)
adds parallelism on top (limited on THIS machine by the FSRS benchmark eating ~6 cores). **NOTE:** the
500+500 export is blocked on `label_filter_db` coverage -- it has only ~417 held-out users (1-420; the
champion trained on 1000-2499), so a fresh held-out range needs `find_equalize_test_reviews` first
(map_size 2GB to stay compatible with the export's 2GB open).

## ⚠ What the throughput numbers MEAN (states/s vs reviews/s; batched-query vs sequential-build)

(Andrew 2026-06-30) The B=128 (17k) and multithread (132k) figures are **batched-QUERY** throughput =
one forward step across B **INDEPENDENT** streams. **State-BUILDING is sequential** at two levels:
(1) per card the WKV recurrence chains (each review's state feeds the next); (2) within a user the
**shared streams** (note/deck/preset/global) are updated in **chronological review order**, so review
N+1 (any card) depends on the shared state after review N. So **one user's history build can't be
batched across its own cards** -> it is B=1 sequential ≈ **408 reviews/s** (the `--bench` full replay).

- **Batched B=128 / MT** legitimately applies to INDEPENDENT streams: building **many users in parallel**
  (the export/sibling case) or **re-scoring many cards read-only** (states already built). Best called
  **states/s** (in a re-score no review is consumed -- each step just computes one card's state-forward).
- **Single-user sequential build** = **reviews/s** (B=1, ~408) -- shared-stream chronological coupling.
  This is where the plain-Rust fast path's **4.84x at B=1** (~2000 states/s) is the deployment-relevant
  win; for Anki the build is one-time + cached, then incremental (1 review/update), and retrievability
  prediction uses the STORED forgetting curve (`predict_ahead`, no forward at all).
- **Terminology:** count = forward steps. Use **states/s** for batched/query/MT (no reviews consumed);
  **reviews/s** for sequential history processing (one review -> one state). Numbers are identical; the
  label reflects whether reviews are being consumed. The 17k/132k headlines are states/s, NOT a single
  user's sequential reviews/s.

## Multithreading (batch/card parallelism) — the big throughput lever

Cards are independent (read-only query) -> embarrassingly parallel. `--bench-mt <secs> <B> <threads>`
spawns `threads` OS threads, each running candle `review_batched` at B (single-thread gemm, RAYON=1) on
shared read-only weights+states; aggregate rev/s. **Measured 2026-06-30** (champ_h2k16, B=128, *with the
FSRS benchmark eating ~5-6 cores* so a clean machine scales higher):

| threads | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| agg rev/s | 15,134 | 31,589 | 63,379 | 101,989 | 130,987 | 132,424 |
| vs 1t | 1.00x | 2.09x | 4.19x | 6.74x | **8.66x** | 8.75x |

Near-linear to 4 threads, saturating ~12 (~**8.7x -> ~132k rev/s**). **B=128 stays optimal under MT** (12t:
B=64=102k, B=128=125k, B=256=62k -> 256 cache-spills across threads). Deployment config: ~12 threads x
B=128. (This is a scaling measurement, not a single-thread protocol iteration -> no Wilcoxon.) The
plain-Rust fast path starts lower at B=128 (10.9k vs 15.1k) so it won't beat candle MT there; it remains
the small-B/sequential engine.

### ★ CAVEAT (Andrew 2026-06-30): 17k/132k are batched *QUERY*, NOT sequential *BUILD* throughput

The B=128 / 17k (1-thread) and 132k (MT) numbers are ONE forward step across 128 **independent** streams.
**State-building is sequential** at two levels: (1) per card the WKV recurrence chains (review N's state
feeds N+1); (2) within a user the SHARED streams (note/deck/preset/global) update in **chronological review
order**, so even different cards of one user can't be batched for the build (the `card->note->deck->preset->
global` hierarchy couples them in time). So **a single user's full-history build is B=1 sequential (~408
rev/s `--bench`), not 17k.** B=128/17k legitimately applies ONLY to *independent* batches: building across
**many users at once** (the export/sibling pipeline) or **read-only re-scoring** of many already-built cards.
For one Anki user: build once (sequential, ~408 rev/s, then cached) + incremental 1 review/update, and
retrievability uses the STORED forgetting curve (`predict_ahead`, no forward) -> so the deployment-relevant
single-user engine is the **B=1 path where the plain-Rust fast path wins 4.84x (~2k rev/s)**, not the
batched-query headline. 17k/132k remain the right numbers for the multi-user export + collection re-scoring.

## Iterations

| # | timestamp | before rev/s | after rev/s | speedup | wilcoxon_p | trials | result | summary |
|---|---|---|---|---|---|---|---|---|
| 0 | 2026-06-30 | — | 15970 | — | — | — | baseline | champ_h2k16 batched B=128 single-thread baseline (16 MB; B=1 replay ~408) |
| 1 | 2026-06-30 | 15872 | 16112 | 1.015x | 1.22e-2 | 20 | rejected | target-cpu=native+LTO+cu1: gemm already runtime-dispatches AVX2/FMA -> no win at B=128 |
| 2 | 2026-06-30 | 16147 | 10923 | 0.68x | n/a | n/a | rejected@B128 | plain-Rust f32 forward (src/fast.rs): parity 3e-7, but naive matmul loses to gemm at B=128 |

**Iter 2 detail — plain-Rust rewrite (`src/fast.rs`, `--verify-fast`/`--bench-synth-fast`):** a full f32
re-implementation of `review_batched` on flat buffers, **parity-verified 3e-7** vs candle (after matching
`GN_EPS=64e-5`). Throughput is **FLAT ~11,800 rev/s** (my matmul is naive O(B), no batch scaling), so it
**WINS BIG at small B** (B=1 **4.84x**, B=4 3.0x, B=16 1.5x, B=32 1.09x) but **loses at B≥64** because
candle's `gemm` crate (blocked SIMD) scales with B and overtakes around **B≈40** (B=128: candle 16147 vs
fast 10923 = 0.68x). => At the B=128 deployment point candle is gemm-bound and already near-optimal; the
rewrite's win is on the **small-B / sequential (export, sibling state-production) path** (B=1 4.84x), NOT
queue-scoring. Kept in-tree (parity-correct, valuable for small-B). To win at B=128 the fast path needs a
gemm-class batched matmul (the `gemm` crate or a blocked micro-kernel). Champion stays candle.

## Addon comparison (rwkv-srs Anki addon vs our CPU inference) -- 2026-06-30

Downloaded `rwkv-srs-anki-dev-cached-windows-x86_64.ankiaddon` (Andrew). It ships its OWN Rust CPU
inference (`vendor/rwkv_srs/_native.pyd`, candle+Rayon) running the **published d=128 / 4-head / 2.76M**
model (`pretrained/RWKV_trained_on_{101_4999,5000_10000}.safetensors`). Its **predict** = `RWKV_SRS.predict`
/ `predict_many(batch_size=192, num_threads=physical_cores)` = immediate recall prob `1 - P(again)`, state
read-only (== our `review_batched`); `process_many` = sequential state build (== our sequential `review`
loop); `get_probability(curve, elapsed)` = the forgetting-curve retention. Same imm task + same arch family
as ours. Its `_native.pyd` imports in our venv 3.12; our dim-agnostic engine loads the addon's d=128
safetensors directly (keys match) -> a true same-model race is possible. Harness in `scratchpad/cpu_compare/`
(`addon_bench.py`, `run_compare.py`); addon review-dicts built from anki-revlogs-10k via the same
`revlogs JOIN cards JOIN decks` merge our export uses (the addon's feature engineering is byte-identical to
ours -- same `RNNProcess.add_same`, `CARD_FEATURE_COLUMNS`, scale_* -- so the workloads are equivalent).

**Two axes (Andrew chose BOTH): (A) same-model engine race -- both engines on d=128 (isolates the engine);
(B) full-stack -- our d=32 H=2/K=16 champion + our engine vs the addon's d=128 + its engine.** Workloads:
sequential build (reviews/s) + batched read-only **predict** -- which outputs a **recall probability R**
(`1 - P(again)`) per card per forward, NOT a state (the predict path is read-only). Predict throughput unit
= **R-predictions/s** (earlier loosely called "states/s"; corrected per Andrew 2026-06-30). Single + 8-thread.

### Results -- user 107 (5229 reviews), B=192, best-of-3 (recovers least-contended slice)
(predict columns = **R-predictions/s**, read-only; build = reviews/s, state-advancing)
| engine / model       | build rev/s | pred 1T (candle) | pred 1T (fast.rs) | pred 8T |
|---|---|---|---|---|
| ADDON d128 (native)  | 270 | 2600 | n/a  | 2932  |
| OURS  d128 (matched) | 235 | 1737 | 1661 | 3723  |
| OURS  d32 champion   | 400 | 14985 | 11525 | 63068 |

- **(A) Same model (d=128):** the addon's native engine is **~1.15x faster build** (270 vs 235) and
  **~1.5x faster single-thread predict** (2600 vs 1737) than our candle path -- their kernel is a bit more
  optimized per-flop. BUT our multithread scales far better (ours 1.7k->3.7k = 2.1x for 8T; addon
  2.6k->2.9k = 1.13x), so at 8 threads **OURS overtakes** (3723 vs 2932, ~1.27x). Our fast.rs loses to
  candle at B=192 (gemm ceiling), as expected. Net: the two engines are in the same league at matched
  model; neither dominates.
- **(B) Full-stack (what we'd ship):** our d=32 champion **beats the addon's d=128 decisively** --
  **1.48x build** (400 vs 270), **5.8x single-thread predict** (14985 vs 2600), **21.5x 8-thread predict**
  (63068 vs 2932). Almost entirely the 14x-smaller model (state + matmuls shrink ~8x) plus better MT scaling.
- **Real-world (single-user recalc, the sequential build is the binding cost):** a 400k-review power user
  (e.g. export user 6197) = champion ~1000 s (~17 min) vs addon-d128 ~1500 s (~25 min); a typical 5k-review
  user = ~13 s vs ~19 s. Predict (batched) is sub-second either way.

**Caveats:** machine ran a permanent FSRS benchmark (~8 cores) + the export during all runs -> absolute
numbers are depressed, but both engines saw matched contention and best-of-3 recovers near-uncontended
single-thread, so the RATIOS hold. Our predict bench uses synthetic warmed states (same shapes as the
addon's real warmed cards -> same compute; RWKV WKV states are dense, no sparsity exploit). Strict numeric
imm parity not bit-checked: the random per-id encodings use different RNG seeds in the addon vs our trace
export, so values differ by design -- irrelevant to speed; the engines are mathematically identical (same
arch + weights, our engine runs the addon's d=128 file). Single representative user; build/predict rates
are ~per-review/per-state constant so they generalize (a large-user pass can confirm if wanted).
