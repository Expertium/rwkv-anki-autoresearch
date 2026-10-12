# CPU inference — the metric the track-2 ablations exist for

**Andrew, 2026-07-25:** *"For state size reduction we have quantization. I told you to do
ablations hoping that fewer params → faster CPU inference in Anki"* and *"Training speed
matters too, but only for us, not for Anki users."* So the scoreboard is:

| metric | who it serves | status |
|---|---|---|
| **CPU rev/s (one review at a time)** | **Anki users — THE product metric** | measured below; Rust number still missing |
| per-entity state bytes | Anki users (disk/sync) | solved by quantization (q72u: 9 B/card) |
| GPU steps/s | us (iteration velocity only) | A0 0.933 → A16 1.746 = 1.87× |
| params | proxy for the two above | 2.76M → 388k = 7.11× |

## Measurement 1 — Python RNN path (`SrsRWKVRnn.review()`), 2026-07-25

`python optimization/cpu_infer_bench.py --threads N --iters M`. fp32, state carried across
calls, median of 200 calls. (Param counts here are the RNN twin's — it does not apply the
GRU-head/strip env flags — but every row runs the same code, so the width scaling is valid.)

| arch | MAC/review | ms/review (3 thr) | rev/s | vs A1 |
|---|---|---|---|---|
| A1 d=128 full | 2,635,776 | 9.82 | 101.8 | 1.00× |
| A9/A13 d=128 | 2,174,976 | 8.29 | 120.6 | 1.18× |
| A14 d=128 lora8 | 2,081,792 | 7.97 | 125.5 | 1.23× |
| A15 d=96 | 1,217,280 | 7.72 | 129.6 | 1.27× |
| A16 d=64 | 582,144 | 7.91 | 126.4 | 1.24× |

**⚠ The headline: a 4.5× cut in arithmetic bought 1.24× wall-clock, and it PLATEAUED after
A14 — the two width cuts (A15, A16) bought essentially nothing.**

### Why: this path is overhead-bound, not compute-bound
- Effective arithmetic rate: **0.30 GMAC/s (A1) and 0.08 GMAC/s (A16)** — a single modern
  core sustains ~5–20 GMAC/s with AVX2/FMA. We are **20–100× below the hardware**, so
  wall-clock is set by per-op Python/dispatch cost, not by multiplies.
- Cost therefore scales with the NUMBER OF SEQUENTIAL OPS ≈ layers × ops-per-layer, which
  the width cuts do not change: A16 runs the same 13 layers as A15, just with smaller
  tensors, so it takes the same time.
- Thread-count control confirms it: **1 thread is FASTEST** (A1 8.82 ms @ 1 thr vs 9.82 @
  3 vs 12.50 @ 6). Tiny tensors + thread sync = negative scaling. **Deploy should run
  single-threaded.**
- Corollary: as the model shrinks, its arithmetic efficiency gets WORSE (0.30 → 0.08
  GMAC/s) because fixed overhead is a growing share.

## What this means for the ablation programme

1. **Param count is a good proxy for state and for training speed, but NOT for CPU
   inference speed in an overhead-bound engine.** Width cuts (A15/A16) reduce FLOPs and
   state; they do not reduce op count.
2. **In an overhead-bound regime the lever is DEPTH and STREAM COUNT** (13 layers × 5
   streams = the sequential op chain), not width. But the depth floors are already mapped
   and accuracy-limited (card 2 / deck 4 / note 1 / preset 3 / user 3 — A10–A12 all failed).
3. **In a compute-bound regime the lever is width** — which is what the Rust engine should
   deliver, since it has ~no per-op interpreter overhead. Historically Rust ran this model
   ~10× faster than the Python path.
4. **So the question "did the ablations buy CPU speed?" cannot be answered until the Rust
   engine runs the track-2 architecture.** That port is the gating work: `rust/rwkv-infer`
   currently implements the track-1 champion shape and does NOT support the GRU curve head,
   `RWKV_STRIP_CMIX`, or arbitrary `d_model`/head counts.

## Next step (queued)

Port the track-2 arch to `rust/rwkv-infer` (GRU head + cmix strips + parameterized
d_model/H), then re-run this ladder through `optimization/measure_throughput.py`, which
already drives the Rust binary on the 3 reference users. Expected outcome if Rust is
compute-bound: the A15/A16 width cuts finally show up as real rev/s, roughly tracking the
4.5× MAC reduction. If Rust turns out to be overhead-bound too, the honest conclusion is
that further width cuts are for state and training only, and CPU speed needs op-fusion work
instead — which is roadmap step 4 (speed) rather than step 5 (params).

## Measurement 2 — RUST engine (`rust/rwkv-infer`, the deploy path), 2026-07-26

**The missing number is no longer missing.** The engine could not run the track-2
architecture at all until today's port steps 1-3 (shape detection, GRU curve head, per-layer
cmix skip + state clamp — commits `1f22e11`, `1620c82`, and this one). Both models below are
driven through the SAME engine, the SAME reference traces (users 107/136/156) and the same
machine, via `./rust/rwkv-infer/target/release/rwkv-infer.exe` with `RWKV_WEIGHTS` set;
default fast path (`fast.rs`), B=1, single-threaded, fp32, no state compression.

| model | params | rev/s | vs baseline |
|---|---|---|---|
| `rwkv_ref_558` — the original 2.76M | 2,762,884 | ~714 | 1.00× |
| **A18 champion** (d=80, GRU head, stripped mixers) | **557,246** | **~1,703** | **2.39×** |

**A 4.96× param cut buys 2.39× throughput here, versus 1.24× and plateauing in the Python
RNN path.** So the ablation programme does pay off for Anki users — but only in the engine
that will actually ship, and still sublinearly (2.39× for 4.96×), consistent with a workload
that is part per-op overhead and part real arithmetic. The Python-path plateau in
Measurement 1 was an artefact of that path's ~0.08-0.30 GMAC/s overhead ceiling, NOT evidence
that width had stopped mattering. Two lessons stand: measure the deploy engine, and do not
generalize a scaling curve from a harness running 20-100× below the hardware's rate.

⚠ **Caveats, stated plainly.** (1) ~~**Parity is NOT established**~~ **RESOLVED LATER THE SAME
DAY — parity PASSES.** This caveat was written before the gate was run and is kept only so the
sequence is legible. The A18 port verified at **imm 0.000035 / ahead 0.000044 against a ±0.0005
tolerance** (14x and 11x inside), once the *reference trace* was regenerated — the old June trace
was not reproducible by current Python, so the gate had been scoring stale artifacts rather than
the engine. Procedure and the `trace_selfcontained.py` diagnostic are in AGENTS.md §11. So the
throughput below is now backed by predictions that are also *correct*, not merely finite.
(2) The two rows differ in more than width — architecture generation, curve head and
stripped sublayers all move together. That is deliberate: it is the honest "what we shipped
before vs what we would ship now" comparison, not a controlled width sweep. (3) Not yet
measured with quantized state; **the PAVA button API is now measured — Measurement 3 below.**

## Measurement 3 — cost of serving the 4 PAVA button intervals, 2026-07-27

The deploy contract serves four counterfactual button predictions per card, so Anki pays this on
**every card it shows** — it sits on the interactive path in a way aggregate rev/s does not.
`--bench-buttons` (A18 champion weights, `fast.rs`, single-threaded, fp32, B=1 randomized per-card
state so the B=1 -> B=4 tiling is genuinely exercised):

| call | ms/call | note |
|---|---|---|
| `review` B=1 | 0.283 | baseline: one plain prediction |
| `review` B=4 | 0.711 | the probe forward alone — exactly `button_intervals`' forward component |
| `button_intervals` | **0.762** | forward + the 50-step bisection |

**Three things follow, and they close the question rather than open it.**
1. **It is cheap in absolute terms: 0.76 ms per card, 2.69x a plain prediction.** At sub-millisecond
   there is no user-visible cost to serving intervals; this is not a deploy risk.
2. **93% of it is the forward and 7% is the solver** (0.711 vs 0.051 ms). So the bisection's step
   count is NOT a lever — 50 steps buys ~1e-15 relative precision on the bracket, absurd overkill
   for an interval in seconds, and cutting it to 25 would save ~0.025 ms of a 0.762 ms call. Don't
   bother. Any real win has to come from the forward, i.e. the same width/SIMD levers as everything
   else.
3. **The 4 probes batched cost 2.5x a B=1 review, not 4x** — batching already recovers ~37% of the
   naive cost, so the existing one-call-of-4 implementation is doing its job and there is no
   cheap restructuring left.

⚠ **Read the RATIOS, not the absolute rate.** This bench reuses pre-built synthetic states and
times a single step, so its B=1 figure (~3,255 rev/s) runs well above Measurement 2's trace-driven
~1,703 rev/s, which includes real state management across thousands of chained reviews. The two
are not comparable and Measurement 2 is the honest throughput number; what Measurement 3 
establishes is the *relative* cost of buttons, where both sides share the identical harness.

## Measurement 4 — what STATE COMPRESSION costs in time, 2026-07-27

Measurements 2-3 are fp32. The config we actually intend to ship is **q72u** — 72 bits/layer, a
**9-byte card state, 256x compression**. Its accuracy price is on the record (+0.00114/+0.00021 vs
fp32); its **time** price was never measured. Same harness as Measurement 3, A18 weights,
single-threaded, so the columns are directly comparable:

| state config | review B=1 | vs fp32 | buttons | vs fp32 |
|---|---|---|---|---|
| fp32, no compression | 0.307 ms | 1.00x | 0.839 ms | 1.00x |
| low-rank rank-1 int4 (card+note) | 0.373 ms | 1.21x | 0.977 ms | 1.16x |
| + per-column scale + WKV PQ codebook | 0.547 ms | 1.78x | 1.770 ms | 2.11x |
| **full q72u (the deploy config)** | **0.917 ms** | **2.99x** | **3.530 ms** | **4.21x** |

**The headline: 256x state compression costs ~3x inference time** (4.2x when also serving buttons).
Both remain comfortable in absolute terms — under 4 ms per card — so this is a Pareto *choice*, not
a blocker: 9 bytes/card and 3x slower, or 51 KiB/card and fast. Worth putting to Andrew explicitly,
because the compression work was justified on SIZE and its speed cost had simply never been on the
table.

**Where the time goes:** the single most expensive component is the **shift PQ + 1-bit norms**
(0.547 -> 0.917 ms, +68% of the fp32 baseline in one step) — unsurprising, since the shift catalog
is m2b12L, i.e. 2 chunks x 4096 entries searched per shift vector per layer per stream, against the
WKV side's single 1024-entry joint catalog. If this tradeoff is ever worth revisiting, the shift
side is where the time is, and the size ladder already showed shift coding is where the *bits* are
too — so it is one lever moving both.

⚠ **WARM SEARCH IS LOAD-BEARING, and getting this wrong doubles the apparent cost.**
`FastLayerState::warm_wkv` / `warm_shift` carry the entity's previously-winning centroid indices and
travel with the state; `fast.rs:585,614` take the warm path only when they are non-empty. The first
version of this bench passed them empty and rebuilt state per iteration, measuring a COLD
full-catalog search and reporting **6.17x** for q72u instead of the true **2.99x**. The bench now
threads returned state back in and warms 32 reviews before timing. Any future harness touching
compressed state must do the same.

⚠ The forward/solver split from Measurement 3 **does not survive heavy quantization** — under q72u
`button_intervals` (3.530 ms) comes out *below* the B=4 review baseline (3.810 ms), i.e. a nonsense
negative solver cost. That is not noise: the 4 probes are near-identical vectors (same row, only the
button one-hot and duration differ) tiled from ONE warm state, so their warm search converges
immediately, whereas the B=4 control holds four independently-random states. Real conclusion, which
is better news than the split would have been: **serving 4 buttons costs far less than 4 independent
reviews under compression**, because the probes are mutually coherent.

## Does the iter-41 interleaved schedule cost anything at deploy? (provisional, 2026-08-11)

**Provisionally no.** The schedule reorders *when* layers run, not *how many* — the champion's
depths `[2,1,4,3,3]` are 13 layer-steps whichever way they are visited, and `interleave_visits()`
emits exactly `sum(depths)` pairs (unit-tested). So the arithmetic per review is identical and only
locality changes: the sequential chain finishes one stream's state before touching the next, while
the interleaved one keeps all five states hot across the whole review.

Observed during the gap-7/8 parity work, on the same machine minutes apart, fast path, B=1:

| model | schedule | rev/s (3 users) |
|---|---|---|
| iter 31 (`architecture_d80_lora4`, depths 2/4/1/3/3) | sequential | 1505, 1490, 1519 |
| iter 41 (`_cnd`, depths 2/1/4/3/3) | interleaved | 1552, 1461, 1533 |

The two are the SAME layer multiset (13), so this is apples-to-apples rather than a size
comparison. The ranges overlap and no difference is resolvable.

⚠ **This is NOT a speed claim under the protocol** — it is six unpaired single runs, not the
20-trial paired Wilcoxon at pinned CPU frequency that §11 requires, and they were taken while a
GPU training chain was resident. Treat it as "no reason to expect a deploy cost", and measure
properly if interleaving ever needs to be defended on speed. The useful half is the structural
argument above, which does not depend on the timings.

## Measurement 5 — ENERGY per user over a lifetime, 2026-08-12

Prompted by a Discord jab ("post metrics for how much electricity RWKV will use ... so we know how
much people are destroying the planet for marginally improved learning"). Worth having a real
answer: it is the question an Anki maintainer would reasonably ask before shipping a neural
scheduler, and both ingredients are now measured rather than guessed.

**Per card shown = 8.51 ms of ONE CPU core** under the shipping config: 1.756 ms state update
(Measurement 2's honest trace-driven 1,703 rev/s, scaled by Measurement 4's 2.99x q72u cost) +
6.759 ms to serve the four button intervals (Measurement 4's buttons/review ratio 3.85x). Review
counts are the full census in `research_5k_notes.md` (all 9,934 users). Marginal power 15 W for one
busy core on a mainstream desktop CPU.

| user | reviews | CPU time | energy | cost @$0.15/kWh | CO2 @400 g/kWh |
|---|---|---|---|---|---|
| median | 31,070 | 4.4 min | **1.10 Wh** | $0.00017 | 0.44 g |
| mean | 72,989 | 10.4 min | 2.59 Wh | $0.00039 | 1.04 g |
| p90 | 180,883 | 25.7 min | 6.42 Wh | $0.00096 | 2.57 g |
| p99 | 543,161 | 77.1 min | 19.27 Wh | $0.00289 | 7.71 g |
| heaviest of 9,934 | 3,910,718 | 9.3 h | 138.7 Wh | $0.021 | 55.5 g |
| hypothetical 100/day x 20 yr | 730,000 | 103.6 min | 25.90 Wh | $0.0039 | 10.4 g |

**A median user's ENTIRE lifetime of inference is 1.10 Wh** = 7.4% of a phone charge, 1.1% of
boiling a litre of water, 1.6% of driving 1 km in an EV, ~half of one LLM chat response.

**★ THE FRAMING THAT SETTLES IT: the model is 0.064% of the energy of the review it serves.** Per
review, RWKV burns **0.128 J**; the device sits on at ~20 W for the ~10 s the human spends reading
and answering = **200 J**. Looking at the card costs **1,566x** more than predicting it. Any
argument that neural scheduling is an environmental cost has to first explain the screen.

**The one-off training is the larger term, and still small.** ~1,000 GPU-hours of research at ~300 W
= 300 kWh = ~120 kg CO2. Amortized: 3.00 Wh/user at 100k users, **0.30 Wh at 1M** (27% of a median
user's lifetime inference), 0.03 Wh at 10M. So all-in, ~1.4 Wh per user at 1M users — ~10% of a
phone charge for a lifetime of scheduling.

⚠ Caveats: measured on this machine (5950X) through the Rust engine, single-threaded — a mainstream
i5 is the same order, not the same number; 15 W marginal is an estimate, not a measurement; it
assumes button intervals are served on EVERY card (an upper bound); and the dataset's review counts
are to-date, not full lifetimes, hence the 20-year hypothetical row. FSRS by comparison is a closed
form costing microseconds — RWKV is ~1000x its compute, which is still ~1000x of nothing.

## PRIOR ART TO READ FIRST WHEN PHASE 7 STARTS: Clanki PR 160 (Andrew 2026-09-19)

https://github.com/Expertium/Clanki/pull/160 (MERGED; `rslib/src/rwkv/mod.rs`, the Anki fork's engine for the OLD d=128 model). Andrew: "record this for when we start working on speeding up CPU inference (not that there is much left to optimize)". What it established, measured on his collection:

* **The AVX2/FMA kernels were ALREADY in the fork** (from JSchoreels, 2026-07-14: `dot_product`, `add_scaled_in_place`, a 4-row projection block, the recurrence update; runtime `is_x86_feature_detected!("avx2")` + `"fma"`, scalar fallback). Our `rust/rwkv-infer` has NO SIMD; the same source is `vendor/jschoreels_anki/rust/x86_simd.patch` (AGPL: flag to Andrew, keep provenance).
* **What still paid: BATCHING rows** (`Linear::apply_batch` -> `apply_block_into`, 32 rows per block; batch = items / threads, rounded up to 4, clamped 4..128): 1,024 real query rows per-row -> batched = **+147%** (548 -> 221 ms, p 1e-34). A **wider projection tile** (4 rows x 16 outputs, 8 accumulators, no per-row zero branch) = **+57%** at kernel level. Same lesson as `BATCHING_PLAN.md`: matrix x matrix beats a faster dot product at our widths.
* **End to end the gain is small**: calibration replay 11.08 -> 10.16 s per 100k reviews; the Total Knowledge graph did NOT move, because its time is in the Python glue per card per day (input building and parsing), not in the Rust query. **Profile the caller before the kernel.**
* **The parity gate that worked**: scalar path = the test reference (bit-exact tests pinned to it; SIMD opt-in per test through a thread-local switch); AVX2 vs scalar max |diff| 9.5e-7 over 100,000 replay predictions (p99 1.2e-7), new vs old AVX2 kernel bit-identical, batched vs per-row 2.4e-7. FMA and summation order are not bit-exact against scalar and that is fine; ~1e-3 per prediction would be a formula error. State stays f32; no crate-wide target feature; NEON untouched; macOS keeps Accelerate.
* An algorithmic lever I gave that session and it has not used yet: a query step does not change state, so one warm state per card serves every day of a per-day graph; only the input projection and the head re-run.


## 2026-09-19 -- PROFILE OF THE DEPLOY-QUANTIZED ENGINE: THE CATALOG SEARCH WAS 85% OF THE TIME; FIXED, PICK-IDENTICAL (speed plan item a)

Measured on `reference_final_inputs` (w10lf WS weights, 103 inputs, users 107 / 136 / 156, ~5,300 reviews each), one thread,
beside a local GPU training and Andrew's FSRS-7 run (so absolute numbers are ceilings on time, the ratios are the result).
Deploy env = lfq1's learned catalogs: `RWKV_STATE_LOWRANK_SCOPE=card:1:int4,note:1:int4 RWKV_LOWRANK_PQ=..wkvcb.. RWKV_SHIFT_PQ=..shiftcb..
RWKV_QUANT_SHIFTS=1 RWKV_PQ_NORM_BITS=1` + `RWKV_STREAM_ORDER=card,note,deck,preset,user RWKV_INTERLEAVE=1`.

| configuration | reviews/s | ms per review |
|---|---|---|
| no state quantization (the number every earlier speed row used) | 1,395 | 0.72 |
| + WKV rank-1 + b10 joint catalog | 654 | 1.53 |
| + shift catalog m2b12 = THE DEPLOY CONTRACT, serial warm scan (before) | 233 | 4.29 |
| the same, prescreen scan (after) | ~757 (610-760 under load) | ~1.32 |
| the same with the b13 WKV catalog | ~440 | ~2.3 |

* **Every earlier CPU speed number in this file is for the UNQUANTIZED engine.** The shipped contract was 6x slower, and ~85% of
  its time was the nearest-centroid search: 50 searches per review (4096 x 40 shift chunks, 1024 x 32 joint WKV entries), done
  by a serial early-exit scalar loop (60 us per search even with the warm bound).
* **The fix (`model.rs::nearest`, default on; `RWKV_PQ_SCAN=exact` restores the serial scan):** stage 1 scores every centroid
  with `||c||^2 - 2 q.c` on a TRANSPOSED catalog (sub_dim axpy passes over ncent floats: vectorizes across centroids, AVX2+FMA
  by runtime detection, plain fallback); stage 2 re-measures every centroid within 1e-3 of the best score (1.01 candidates on
  average) with the EXACT serial sum and the same tie rule. The winner is provably in the candidate set while the f32 error of
  a <= 64-term unit-scale sum (~1e-5) stays under 5e-4, so the pick -- and therefore every prediction -- is identical on every
  CPU, whatever the SIMD path. **Verified: the three users' prediction files are BYTE-IDENTICAL to the serial scan's.** 8.4 us
  per search (7x), 3.25x end to end. A first version with a per-centroid 8-lane dot product was latency-bound (60 us): the
  transposed layout is what pays.
* The warm index is still stored (state layout unchanged) but no longer needed for speed.
* What is left of the quantization overhead (~0.6 ms per review): ~0.42 ms of search, the rest = the rank-1 power iteration.
  Next candidates, unmeasured: skip the search when the vector did not change (query rows), SIMD for the network's own
  matvecs (jschoreels' AVX2 patch, AGPL -- ask first), exp / sigmoid approximations (~2,500 calls per review).
* The Anki port must carry this search (or the serial one); both give the same indices, so stored states are interchangeable.

* **Later the same day: ~990 rev/s (4.2x), still byte-identical** -- the query pass no longer quantizes a state it discards (`review_query`), and the rank-1 power iteration runs on stack matrices. Tables (states/s, R/s, quantization cost by component): `DEPLOY_SPEED_LOG.md` "2026-09-19".

## 2026-09-19 -- do quantized states diverge across CPUs / kernel versions? (question from the Clanki session)

Concern: SIMD and scalar kernels differ by ~1e-6; with QUANTIZED stored states a near-tie in the catalog search or an int4
rounding can flip, so the same card gets a different state on another CPU. MEASURED with `RWKV_DEBUG_JITTER=<rel>` (debug only,
`model.rs::compress_wkv_state`: every state element times 1 +/- rel, deterministic hash sign, before compression; unset = preds
byte-identical), deploy-quantized engine, users 107 / 136 / 156 of `reference_final_inputs` (16,033 imm / 13,314 ahead predictions):

| jitter | mean abs dp imm | p99 | share > 1e-3 | share > 1e-2 | max | identical |
|---|---|---|---|---|---|---|
| 1e-7 | 2.5e-04 | 4.8e-03 | 2.5% | 0.51% | 0.202 | 12% |
| 1e-6 | 7.5e-05 | 1.3e-03 | 1.5% | 0.08% | 0.062 | 30% |
| 1e-5 | 1.6e-04 | 3.6e-03 | 2.8% | 0.29% | 0.055 | 29% |

* NOT dose-dependent: any perturbation flips some picks; the size is chaotic, not proportional. The signed mean is ~0 (3e-05 at
  most), so accuracy does not move; the quantization tax itself is ~0.002-0.003 LogLoss.
* It GROWS along a user's history (user 107, quartile means 2e-05, 1e-04, 4e-05, 6e-04): a flipped card/note state feeds the
  fp32 deck / preset / user recurrences, which never re-quantize.
* Rounding before the quantizer does NOT fix it (rounding has its own boundaries). "Scalar for the state step" means scalar
  for the WHOLE state pass, because k / v / w come out of the network's matvecs. The clean fix is ONE canonical float order:
  the SIMD kernels and the scalar fallback use the same lane count and reduction order with fused multiply-add
  (`f32::mul_add` in the fallback), so every CPU gives the same bits; plus a kernel-version tag on the state cache so a
  kernel change triggers a rebuild from the review log. Query passes store nothing and need no such care.

## 2026-09-20 -- THE ENGINE IS BATCH-1 PER REVIEW, AND A CHUNKED REPLAY IS THE BIGGER LEVER (from the Clanki session)

The Clanki session measured its own bulk replay against ours and the comparison is worth keeping, because
it names a structural gap that none of yesterday's four micro-optimizations touch.

| pass | ms per review | what it contains |
|---|---|---|
| ours, `rwkv-infer` replay, unquantized | 0.46 | query pass + state pass, 558k params, single thread, features PRE-EXPORTED in the trace |
| ours, deploy contract (b10 + m2b12 + 1-bit norms) | 0.68-0.76 | the same, with state quantization (~45% of it) |
| theirs, release build, both lanes + heads + recording | 0.136 | 2.76M params, feature building INSIDE the loop |
| theirs, state pass only, no recorders | 0.064 | the same minus the query lane |

Their `warm_up_reviews_bulk_impl` runs **chunks of 16,384 rows as a wavefront**, so every dispatch and
every weight-matrix read is amortized over thousands of rows; ours steps one review at a time. That is
the whole of the apparent 3.4x-at-5x-the-arithmetic gap, and it is not an implementation trick.

⚠ **Two caveats before quoting their number**, both raised with them and not yet answered: their bulk path
uses `rayon::join` and `into_par_iter`, so 0.136 ms may be WALL-CLOCK over several cores while ours is one
thread -- and Andrew's mobile constraint makes CPU-SECONDS the number, not wall-clock; and it is not stated
whether states are compressed in that pass.

**What it means for us.** In the LIVE Anki scheduler B=1 is inherent -- one review happens at a time, and
0.46 ms is far below any latency a user can feel -- so this does NOT say the shipped path is slow. It says
every BULK use of the engine (a historical replay, a state-cache rebuild, our own speed benchmarks, a future
parity sweep) should go through a chunked/batched path, and that `review_batched` is the piece to grow
rather than another micro-optimization of the B=1 loop.

One number of theirs transfers directly to a question this file asked: with the query lane stubbed out their
recording pass costs 0.071 ms against 0.136 ms, so **the query (prediction) lane is 82% of the recording
cost** -- the second lane really is ~one full lane, exactly as the structure predicts, and there is no
cheaper exact substitute for it.

## 2026-09-25 -- Clanki speed hunt round 5 (fork-side study latency; ON HOLD with all CPU work)

Reported by the Clanki session; nothing measured or changed here. For when CPU work resumes (after the model is
finished, Andrew 2026-09-24):
1. Show answer spends ~4.5 ms predicting the RWKV-Curve button intervals when they are not cached. Predicting them
   while the question shows (Clanki plan item A2) removes it from the visible path.
2. `record_reviewer_answer` adds 3.5 ms to a 7 ms answer op (the Rust answer itself is 2.3 ms).
3. Undo's queue rebuild spends 5.5 ms gathering RWKV keys, 5.0 ms of it reading 2,340 card rows from SQLite.
None of the three is the engine's forward pass; they are call placement and I/O in the fork.

## QUEUED (postponed: Andrew 2026-09-24, CPU work waits for the finished model) -- post-sync refresh in the Anki fork (Clanki hand-over 2026-09-25)
Measured by Clanki's batch-B agent on a copy of Andrew's profile (868k revlog rows, 200 reviews synced from another client; probe `C:\Users\Andrew\clanki-logs\bench\sync6\post_sync_probe.py`, Clanki `138f08fe7`): sync 266 ms, then `refresh_rwkv_state_after_sync` 2,676 ms = checkpoint restore 1,447 ms (`rslib/src/rwkv/mod.rs:1542`) + review fingerprint 880 ms (`scheduler/rwkv.rs`) + input build 215 ms + replay 64 ms.
Ideas to judge when the work reopens: (1) when every synced review is newer than the resident state's last review, keep the resident state and replay only the new reviews (skips the restore; needs a proof that the append-only case gives the same state as restore + replay, e.g. deploy_vs_lmdb-style equality on the final state); (2) otherwise run the restore and the fingerprint at the same time (both release the GIL), about -0.9 s.

## 2026-09-26 -- exact inference speedups: what the Anki fork ALREADY does (Clanki, read from origin/main, not measured)
Do not re-propose these for the fork:
1. Batched query passes across cards: DONE (`review_retrievability_query_refs_batched` -> `run_query_batch`, pinned to the scalar path by `batched_retrievability_matches_scalar_query_path`; batched interval and curve paths too).
2. PAVA button probes only when needed: DONE (R uses the query pass only; the probes run in the interval prediction at show-answer time, ~4.5 ms uncached). Open idea there: predict the intervals while the question shows.
4. Curve cached after each review: DONE for RWKV-Curve (Reschedule, due search, R review orders, Stats Stability / prop:s read the stored curve). RWKV-Instant has no curve and still needs query passes.
Still open: 3. the 4-stage catalog search of the next deploy model (112 recipe) -- measure its share when it ships. All in Clanki's backlog "after the model is final".

## QUEUED (postponed with all CPU work) -- Clanki speed hunt round 8, 2026-09-28 (reported, nothing changed)
1. **First Stats open of a session waits ~0.94 s** in `RwkvInference.card_curve_s90s`: the `OnceLock` in `stored_curve_s90` computes the S90 of all ~159k stored curves on first use (later opens 4-9 ms); every state reload (e.g. after a sync) re-arms it. Fix options: compute the S90s on a background thread right after the state loads, or persist them with the saved state (they are a pure function of the stored curve, so a stale-cache risk exists only if the curve format changes -- key them on the model/kernel version tag). Probe: `clanki-logs/bench/hunt_r8_8.log` r0/r1.
2. **Kept state + exact rebuild at open: the first Stats open froze the main thread 1.45 s** in `deckoptions._warm -> get_current_deck`, while `total_knowledge.start_rwkv`'s `find_cards` took 3.5 s; no Python backend call held the collection. Most likely holder: the rebuild itself, holding the Rust backend's collection lock (the Python side only sees the wait). Check first whether the exact-rebuild loop takes the collection mutex for its whole duration instead of per batch. Probe: `bench/hunt_r8_7.log/.json` r0.
