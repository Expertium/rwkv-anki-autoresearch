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
the engine. Procedure and the `trace_selfcontained.py` diagnostic are in CLAUDE.md §11. So the
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
