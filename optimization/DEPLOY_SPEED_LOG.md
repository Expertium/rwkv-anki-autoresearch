# `review_features` speed log — the deploy CPU scoreboard

One table, tracking CPU throughput of the five-stream state computation. The target function and
its exact signature are recorded in [`DEPLOY_FUNCTIONS.md`](DEPLOY_FUNCTIONS.md); our engine's
counterparts are `Model::review` (candle) and `FastModel::review_batched` (the default fast path).

**Only `review_features` is tracked** (Andrew, 2026-07-26). `retrievability_head` is a
*sub-computation* of it — `review_features` computes retrievability inline — so a second table
would measure part of the same work twice and the two rates could never be summed into a per-review
total. It stays documented in `DEPLOY_FUNCTIONS.md` for reference, not scored here.

**Related, different, do not conflate:** [`cpu_speed_log.md`](cpu_speed_log.md) is the older
append-only log for the same engine, but its unit of pairing is a **simultaneous trial** at B=128
(champion binary and candidate binary launched at the same instant, 1 thread each) — see its
"Protocol (Andrew, 2026-06-30)". [`CPU_INFERENCE.md`](CPU_INFERENCE.md) holds the two
cross-*architecture* measurements (Python RNN path, and the Rust 2.39× at 4.96× fewer params).
This file is per-*optimization*, paired across **users**.

## Protocol

- **Metric: states/s** = `review_features`-equivalent calls per second. One call consumes one
  review and yields all five stream states, so calls/s is the natural unit (the convention in
  `cpu_speed_log.md` §"What the throughput numbers MEAN").
- **Unit of pairing: a USER.** 200 users, each contributing one paired point
  `(before_rate_u, after_rate_u)`. ⚠ This differs from `cpu_speed_log.md` and from CLAUDE.md §11,
  which pair *trials*; both designs are valid, but the p-values are not comparable across them.
- **User set: 5001–5200** (the first 200 of the val half). Speed is accuracy-neutral so any fixed
  set works; using val keeps TEST (7501–10000) untouched per the live rule. The
  "TUNE-EVAL SUBSET OVERFIT" lesson does **not** apply here: these 200 users carry *equality
  assertions*, not accuracy rankings.
- **Median relative speedup** = median over the 200 users of `after_rate_u / before_rate_u`.
  Median, not mean — per-user rates are heavy-tailed in review count.
- **Wilcoxon signed-rank**, one-sided, on the 200 paired rates. Accept a speedup claim only at
  **p < 0.01**.
- **Before/after in one run, same process conditions.** Lock CPU frequency first (CLAUDE.md §11
  has the `powercfg` recipe) and keep the machine otherwise idle — in particular no co-tenant GPU
  or FSRS-benchmark load, which has skewed measurements here before.

## The two assertions every row must carry

1. **`size` identical.** Per-user equalized review count must match iter 0 EXACTLY. It is a
   property of the data and the filters, so any change at all is a pipeline bug, not a result.
2. **LogLoss within ±0.0005 of iter 0**, both modes (ahead and imm), by-user mean.

⚠ **±0.0005 is a CEILING, not an allowance.** These are pure-speed changes: the arithmetic is
supposed to be identical, so **the expected delta is 0.000000**. A row showing +0.0003 has not
"passed with margin" — it has quietly changed the model and needs explaining before it is kept.
Treat any nonzero drift as a bug report. (Contrast the +0.0015 *efficiency* budget in CLAUDE.md §5,
which exists for param-cutting and quantization — changes that are *meant* to cost accuracy.)

## Table — `review_features` (states/s)

Iteration 0 is the baseline: the current engine at the commit named in the row, no optimization
applied. **No rows yet** — the CPU optimization work is deliberately queued behind the remaining
algorithmic improvements and the new input features (Andrew, 2026-07-26). Do not backfill this
table from `CPU_INFERENCE.md`: those are cross-architecture numbers, measured without per-user
pairing, and would not satisfy the columns below.

| iter | change | states/s before | states/s after | median rel. speedup (200 u) | Wilcoxon p | size identical | Δ LogLoss vs iter 0 (ahead / imm) | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline — engine at `a3f7003`, parity-verified | — | *(to measure)* | 1.00× | n/a | n/a (defines it) | 0.000000 / 0.000000 | baseline |

### Candidate optimizations, roughly by expected value

From `rust/rwkv-infer/TRACK2_PORT_PLAN.md` §"Order of work" step 6:

1. **AVX2/FMA `dot_product` + `add_scaled_in_place`** — we have no SIMD at all. ⚠ The reference
   implementation is AGPL (`vendor/jschoreels_anki/rust/x86_simd.patch`); these are ~30 lines of
   standard intrinsics we can also write independently. See `DEPLOY_FUNCTIONS.md` §Licensing.
2. **Batched x86 GEMM path** — the fork routes per-review matvecs through `cblas_sgemm` on macOS
   but falls back to scalar on x86. Unclaimed, and the direct fix for an overhead-bound profile.
3. **Single-threaded by default** — already measured: 1 thread beats 3 and 6 on this workload.

Note that the 2.39× already recorded in `CPU_INFERENCE.md` was measured with **no explicit SIMD on
either side** (LLVM auto-vectorization only, equally available to both). Once explicit AVX2 lands,
that cross-architecture ratio may move, because the two models have different head widths
(K=32 vs K=16) and need not vectorize equally well.
