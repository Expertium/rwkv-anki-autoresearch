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
  `(before_rate_u, after_rate_u)`. ⚠ This differs from `cpu_speed_log.md` and from AGENTS.md §11,
  which pair *trials*; both designs are valid, but the p-values are not comparable across them.
- **User set: 5001–5200** (the first 200 of the val half). Speed is accuracy-neutral so any fixed
  set works; using val keeps TEST (7501–10000) untouched per the live rule. The
  "TUNE-EVAL SUBSET OVERFIT" lesson does **not** apply here: these 200 users carry *equality
  assertions*, not accuracy rankings.
- **Median relative speedup** = median over the 200 users of `after_rate_u / before_rate_u`.
  Median, not mean — per-user rates are heavy-tailed in review count.
- **Wilcoxon signed-rank**, one-sided, on the 200 paired rates. Threshold and the rest of the gate:
  see ACCEPTANCE CRITERIA below.
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
measurements here before. This mirrors AGENTS.md §11's "paired simultaneous trial", with the pairing
unit changed from a trial to a user (see above): for each of the 200 users, before and after run
side by side and yield that user's one paired point.

### M3. Pin each worker to its own thread (CPU affinity)

Andrew, 2026-07-26, from the FSRS-7 param-optimization work: pinning **did not reduce noise, but
made runs a few % faster**, and it cannot make noise worse — so do it. The mechanism is cache
warmth: an unpinned worker gets migrated between cores by the Windows scheduler and abandons its
warm L1/L2 each time.

```powershell
# after launching a worker, pin it to ONE logical CPU (bit i = logical CPU i)
(Get-Process -Id $procId).ProcessorAffinity = [IntPtr](1 -shl $cpuIndex)
```

⚠ **Use distinct PHYSICAL cores, not adjacent logical CPUs.** The 5950X is 16 cores / 32 threads,
and Windows numbers SMT siblings adjacently — logical 0 and 1 are two threads of the *same*
physical core. Pinning two workers there makes them fight over one core's execution units. Use
**even** logical indices (0, 2, 4, …) to get one worker per physical core.

**This is where M3 and M2 interact, and getting it wrong silently corrupts the ratio:** the
simultaneous champion and candidate must land on **different physical cores**. If they end up on
SMT siblings they contend with each other, and the measured "speedup" is really a measurement of
that contention. Workers are single-threaded here anyway (`CPU_INFERENCE.md`: 1 thread beats 3
and 6 on this workload), so one worker per physical core is the natural layout.

### M4. LPT — dispatch the largest users first

Sort the 200 users by review count **descending** and hand them to workers in that order (Longest
Processing Time first).

**Why:** it is the makespan that costs wall-clock, not the total work. If a 5,000-review user
happens to be dispatched last, every other worker sits idle while it finishes alone. LPT is the
classic greedy fix and bounds the makespan at ≤ (4/3 − 1/(3m)) × optimal. It pays off here in
particular because our per-user review counts are **heavy-tailed** — the same property that makes
us report a median rather than a mean speedup.

Two invariants:

- **LPT changes only dispatch ORDER, never the work**, so it cannot move LogLoss or `size`. If a
  row's numbers shift when LPT is switched on, that is a bug in the harness, not a scheduling
  effect.
- **Champion and candidate must use the SAME order.** They are paired per user, so a different
  schedule on each side would compare users measured under different machine conditions and quietly
  break the pairing that M2 exists to protect.

## ACCEPTANCE CRITERIA (Andrew, 2026-07-26) — accept a speedup iff ALL FOUR hold

1. **Wilcoxon signed-rank p < 0.01**, one-sided, on the 200 paired per-user rates.
   *Deliberately looser than the accuracy gate's p < 0.0001* (AGENTS.md §"ACCEPTANCE GATE"):
   **timing measurements are noisy**, so demanding accuracy-grade significance from them would
   reject real wins for reasons that have nothing to do with the code.
2. **`size` identical to iter 0.** Per-user equalized review count must match EXACTLY. It is a
   property of the data and the filters, so any change at all is a pipeline bug, not a result.
3. **LogLoss within ±0.0005 of iter 0 on BOTH heads** — ahead (curve head) and imm (rating head),
   by-user mean. See the exact/inexact note below for how to read this one.
4. **Median relative speedup ≥ 1.03 (at least +3%).** A floor on *practical* significance, not
   statistical: with 200 paired users measured simultaneously, the test is sensitive enough to
   certify a 0.5% win with a tiny p-value, and a 0.5% win is not worth the complexity it costs to
   maintain. Criterion 1 says "the speedup is real"; criterion 4 says "it is worth having".

Criteria 1 and 4 are independent and BOTH bind: a large median speedup with an inconsistent sign
across users fails 1, and a rock-solid 1% win fails 4.

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

(Distinct from the +0.0015 *efficiency* budget in AGENTS.md §5, which covers param-cutting and
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

`kind` = **exact** or **inexact** (see criterion 3) — an inexact row must also name the
approximation it made. `verdict` = **accepted** only when all four criteria hold; otherwise
**rejected**, with the failing criterion number. Log rejected rows too: a 2% win that failed the
3% floor is exactly the thing someone will otherwise retry in three weeks.

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

---

# 2026-09-19 -- SINGLE-USER REPLAY TABLES (Andrew: "you can make different tables: states/s, R/s, some table for quantization speed")

These are NOT the 200-user paired-Wilcoxon protocol above (that table stays the acceptance scoreboard). They are
quick replay measurements, one thread, B = 1, on the three parity users of `reference_final_inputs` (107 / 136 / 156,
~5,300 reviews each; w10lf WS weights, 103 inputs, 561,732 params), taken BESIDE a local GPU training and Andrew's
FSRS-7 run -- so read ratios, and treat the absolute rates as floors. The replay binary now prints both rates per
user (`[speed]` line on stderr): the QUERY pass (`review_query`, state read-only = the retrievability / imm
prediction) and the STATE pass (`review_batched` = `review_features`, returns the five new states, quantized when the
deploy env is set). Every change below left the three prediction files BYTE-IDENTICAL to the serial-scan engine's.
Deploy env: `RWKV_STATE_LOWRANK_SCOPE=card:1:int4,note:1:int4 RWKV_LOWRANK_PQ=<wkv catalog> RWKV_SHIFT_PQ=<shift catalog>
RWKV_QUANT_SHIFTS=1 RWKV_PQ_NORM_BITS=1` (+ stream order / interleave; catalogs = lfq1's learned pair).

## Table S -- states/s (the state pass, `review_features`)

| engine state | state quantization | states/s | ms per state |
|---|---|---|---|
| any | off (what every earlier speed number measured) | ~2,800 | 0.36 |
| before 2026-09-19 (serial warm scan) | deploy contract, b10 | ~470 | 2.12 |
| after (prescreen scan + stack power iteration) | deploy contract, b10 | **~1,470** | **0.68** |
| after | deploy contract, b13 WKV catalog (lfq1b13's, if adopted) | ~920 | 1.09 |
| + no-String weight lookup (2026-09-19 evening: FNV-hashed keys, a layer's prefix hashed once; preds byte-identical) | off | ~4,000 | 0.25 |
| + no-String weight lookup | deploy contract, b10 | **~1,915** | **0.52** |
| + 4-row matvec (serial add order kept, AVX2 elementwise, no FMA; byte-identical) | off | ~4,300 | 0.23 |
| + 4-row matvec | deploy contract, b10 | **~1,950** | **0.51** (quantization is now ~0.28 of it, more than the network) |
| + INTEGER prescreen (13-bit catalog + query, `madd_epi16`, rigorous error bound -> 1.05 exact rechecks per search; vectorized min; byte-identical preds) | deploy contract, b10 | **~2,350** | **0.43** (network 0.23 + quantization ~0.20) |

## Table R -- R/s (the query pass: retrievability = 1 - P(Again), and each button probe)

| engine state | R/s | note |
|---|---|---|
| before 2026-09-19, deploy quantization on | ~470 | the query pass QUANTIZED a state it then threw away |
| after | **~2,780** | `review_query` skips the compression (its outputs never depended on it); = the unquantized rate |
| + no-String weight lookup (every `linear` / norm call built 1-2 Strings and hashed them with SipHash: ~20% of the pass) | **~3,970** | byte-identical preds, 3 users; replay 1,395 -> ~1,990 rev/s unquantized, ~990 -> ~1,275 quantized |
| + 4-row matvec | **~4,320** | replay ~2,160 rev/s unquantized, ~1,320 quantized; an AVX2 build of the WKV step gave nothing (K = 16) |

Button intervals = 4 probe query passes (`button_intervals` now uses `review_query` too) -> ~1,000 cards/s today;
R2 (`scratchpad/r2/DESIGN.md`) would make it one query pass.

## Table Q -- what the state quantization costs per state pass (ms, b10 unless noted)

| component | before | after | how |
|---|---|---|---|
| network, no quantization | 0.36 | 0.36 | -- |
| shift catalog search (2 roles x 2 chunks per layer, 4096 x 40) | ~1.3 | ~0.18 | transposed-catalog dot-product prescreen + exact recheck of the ~1.01 near-best centroids |
| WKV joint catalog search (per head, 1024 x 32) | ~0.3 | ~0.06 | the same (b13 = 8192 x 32: ~0.45) |
| rank-1 power iteration (per head, <= 64 iterations) | ~0.14 | ~0.08 | nalgebra STACK matrices for K = 8 / 16 / 32 (same kernels, no heap traffic) |
| **total overhead over the network** | **~1.76** | **~0.32** | |
| the same after the integer prescreen (2026-09-19 evening) | | **~0.20** | integer sums are exact: the prescreen gives the same numbers on every CPU and in the scalar fallback; `RWKV_PQ_SCAN=f32` = the float prescreen, `=exact` = the serial scan |
| replay rate, query + state pass per review (rev/s) | 233 | **~990** | 4.2x; unquantized = 1,396 |

Tried and DROPPED (exact, but slower than the prescreen): a PCA lower-bound scan (project on the top r catalog
directions, skip centroids whose bound exceeds the incumbent). The catalogs are near-isotropic, so r = 16 still left
hundreds of survivors per search: 260 / 290 / 340 / 410 rev/s at r = 4 / 8 / 12 / 16 against ~750 for the prescreen.
Left on the table: skipping the search when a shift vector did not
change, SIMD for the network's own matvecs.
