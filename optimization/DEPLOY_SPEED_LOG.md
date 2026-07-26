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
### M1. Lock the CPU frequency BEFORE measuring (once per session, needs admin)

Without this the CPU boosts and throttles on its own schedule and the measurement drifts under
you — a 3.4 GHz baseline that quietly becomes 4.2 GHz mid-run invents a speedup that is not there.
Run in an **elevated PowerShell**:

```powershell
powercfg -attributes SUB_PROCESSOR 75b0ae3f-bce0-45a7-8c89-c9611c25e100 -ATTRIB_HIDE
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCFREQMAX 3400
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
powercfg /setactive SCHEME_CURRENT
```

The first line unhides the max-frequency setting, which Windows keeps hidden by default. Pinning
`PROCTHROTTLEMIN = PROCTHROTTLEMAX = 100` holds the perf state flat. ⚠ `PROCFREQMIN` is **not** a
valid alias — pin the perf state instead, as above. Restore afterwards:

```powershell
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCFREQMAX 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 5
powercfg /setactive SCHEME_CURRENT
```

### M2. ALWAYS run before and after SIMULTANEOUSLY, on different threads

Never measure the champion, then the candidate. Launch **both at the same instant**, each pinned to
its own thread(s), both looping the same frozen work.

**Why:** any outside factor — the FSRS benchmark, a Windows update, thermal drift, another
process waking up — then lands on *both* sides at once and cancels in the ratio. Sequential
measurement silently attributes whatever changed between the two runs to the code change, and on
this machine there is nearly always something else running (Andrew's FSRS benchmark, a Reddit bot,
the liveplot). Simultaneity does not remove the noise; it makes the **sign** of the difference
clean, which is all the Wilcoxon test needs.

Keep the machine otherwise idle anyway — in particular **no co-tenant GPU work**, which has skewed
measurements here before. This mirrors CLAUDE.md §11's "paired simultaneous trial", with the pairing
unit changed from a trial to a user (see above): for each of the 200 users, before and after run
side by side and yield that user's one paired point.

## The two assertions every row must carry

1. **`size` identical.** Per-user equalized review count must match iter 0 EXACTLY. It is a
   property of the data and the filters, so any change at all is a pipeline bug, not a result.
2. **LogLoss within ±0.0005 of iter 0**, both modes (ahead and imm), by-user mean.

**What the ±0.0005 is for (Andrew, 2026-07-26): it is headroom for INEXACT speedups** — changes
that replace something with a cheaper approximation (a fast `exp`/`tanh`, a lower-precision
accumulation, a truncated softmax, a skipped negligible term). Those genuinely move the numbers a
little, and the band is what says "little enough to keep".

So label every row **exact** or **inexact**, because the same ΔLogLoss means different things:

- **exact** (SIMD, batching, memory layout, allocation removal — the arithmetic is unchanged):
  the expected delta is **0.000000**. Anything else is a bug, not a pass. A nonzero drift on a row
  claiming exactness means the rewrite changed the math — investigate before keeping it.
- **inexact** (a cheaper approximation): spend the band deliberately, and record *what*
  approximation bought the speed. Cost is cumulative against iter 0, not against the previous row,
  so the budget cannot be laundered by taking it 0.0004 at a time.

(Distinct from the +0.0015 *efficiency* budget in CLAUDE.md §5, which covers param-cutting and
quantization — architecture changes rather than implementation changes.)

## Table — `review_features` (states/s)

Iteration 0 is the baseline: the current engine at the commit named in the row, no optimization
applied. **No rows yet** — the CPU optimization work is deliberately queued behind the remaining
algorithmic improvements and the new input features (Andrew, 2026-07-26). Do not backfill this
table from `CPU_INFERENCE.md`: those are cross-architecture numbers, measured without per-user
pairing, and would not satisfy the columns below.

| iter | change | kind | states/s before | states/s after | median rel. speedup (200 u) | Wilcoxon p | size identical | Δ LogLoss vs iter 0 (ahead / imm) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline — engine at `a3f7003`, parity-verified | — | — | *(to measure)* | 1.00× | n/a | n/a (defines it) | 0.000000 / 0.000000 | baseline |

`kind` = **exact** or **inexact** (see the assertions above) — an inexact row must also name the
approximation it made.

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
