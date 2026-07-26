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

⚠ **Caveats, stated plainly.** (1) **Parity is NOT established** — see the RED gate in
`TRACK2_PORT_PLAN.md` §"Order of work" step 4; these predictions are unverified against
Python, though all 5,229+4,215 of them are finite and in (0,1). Throughput is trustworthy
regardless (the same work is done either way); the *correctness* of that work is not yet
proven. (2) The two rows differ in more than width — architecture generation, curve head and
stripped sublayers all move together. That is deliberate: it is the honest "what we shipped
before vs what we would ship now" comparison, not a controlled width sweep. (3) Not yet
measured with quantized state, nor with the PAVA button API (4 extra forwards per press).
