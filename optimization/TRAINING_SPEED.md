# Training speed at d=80 — where the time actually goes

Measured 2026-07-27 on the **current iter-31 / A18 trunk** (d=80, MAX=32768, probe density 0.08,
Muon on, PAVA on), after Andrew asked to speed up training and to "make sure it also profiles
PAVA rectification and anything else that could be slow".

Supersedes the 2026-07-03 profile, which was taken at **d=32 / MAX=110000** and is two
architectures stale. Tooling: `RWKV_PROFILE_STEP` + `RWKV_PROFILE_COUNT` (train_rwkv.py), which
now records **CPU activity as well as CUDA** — that change is what made the answer visible.

## The headline: the step is CPU-DISPATCH-BOUND, not kernel-bound

| | per step |
|---|---|
| **wall clock** | **~1,450-1,540 ms** |
| GPU kernel time (sum of self-times) | **237 ms** (~16%) |
| self CPU time | **915 ms** |
| op dispatches | **90,576** |
| fetch wait | 2.3 ms (0.1% — the loader is NOT a lever, re-confirmed) |

**~85% of training time is not GPU kernel execution.** The GPU is starved while the CPU issues
work: `cudaLaunchKernel` alone is 17,346 calls and **199 ms/step of pure launch overhead**, plus
`cuLaunchKernel` 3,499 calls / 48 ms.

**This explains the iter-33 observation** that halving `MAX_TRAIN_GLOBAL_LEN` cost 2.83x per step
on only 1.13x more rows: at small batch the same ~90k dispatches are amortized over less work.
Cost tracks *parallelism*, not work volume, because dispatch is the bottleneck.

**=> Making kernels faster is close to pointless. The wins are FEWER, BIGGER ops.** That also
retro-justifies why the tensor-core and chunked-matmul rewrites were dead ends: both attack
kernel efficiency, which is 16% of the problem.

### GPU-side split (for completeness — this is the 237 ms)

| share | ms/step | bucket |
|---|---|---|
| 60.09% | 142.72 | other (elementwise/reduce/copy/optim) |
| 29.54% | 70.16 | wkv plain recurrence |
| 10.13% | 24.05 | gemm (linear layers) |
| 0.25% | 0.59 | wkv scan (matmul) |

The d=32 profile said 78 / 18 / 5 — WKV is a materially bigger share at d=80.
⚠ Second-largest single kernel is **`indexing_backward_kernel` at 13.53% = 32 ms/step**, i.e. a
REMAINING deterministic-indexing tax. The 2026-07-03 work (PermGather + flat-row
time_shift_gather) cut the det tax to ~57 ms at d=32; a third gather site evidently survives.

## ★ PAVA IS EXONERATED — it is not slow

It was the prime suspect and the measurement says no:
- `pava_rectify` dispatches **443 aten ops per call**, independent of M (measured at M=1,000 and
  M=100,000, `scratchpad/profile_prep/bench_pava.py`) — but 443 of **90,576** dispatches/step is
  **~1%**, forward and backward together ~2%.
- Its 6 forced device syncs per call are real (the early `break` at `pava.py:92` is dead code —
  6/6 back-merge iterations ran at every pooling rate tested, because `merge.any()` reduces over
  the whole batch), but **all** `cudaStreamSynchronize` in the entire step totals 143 calls /
  **17 ms (~1%)**.

Deleting the dead `break` is still correct and free (with `merge` all-False the body's
`torch.where(upd, ...)` is a no-op, so it is bit-exact), but it is a tidy-up, not a speedup.
**Do not spend effort optimizing PAVA.**

## Ruled out: `empty_cache`

Suspected because `RWKV_EMPTY_CACHE_WINDOW=0` makes d=80 runs call `torch.cuda.empty_cache()` on
**every step for the whole run**, and the code prices it at ~150 ms/step (measured at d=32).
Direct A/B, same config, 50 timed steps after 40 warmup:

| | steps/s | peak reserved |
|---|---|---|
| `EMPTY_CACHE_EVERY=1` (current) | 0.6484 | 8.875 GB |
| `EMPTY_CACHE_EVERY=0` | 0.6887 | 9.007 GB |

**1.06x — ~90 ms/step.** Consistent with the profile's `cudaFree` 299 calls / 102 ms. Not worth
touching: it guards a documented 4x WDDM-paging failure past step ~1000, and a 90-step bench
cannot see that failure. If it is ever wanted, clear every N steps rather than never.

## Ranked opportunities

1. **★ Batch Muon's Newton-Schulz — the best-sized concrete win.**
   `zeropower_via_newtonschulz5` runs **per parameter in a Python loop** (`muon.py:80`), 5
   iterations x 3 matmuls = 15 matmuls per matrix param, so **2,658 `aten::mm`/step costing 92 ms
   of CPU dispatch to do 21.6 ms of GPU work (4.3x)**, plus 30 ms in the optimizer step itself.
   2,658 / 15 ~= 177 matrix parameters, each dispatched individually.
   **Fix:** group params by shape, stack, and run the iteration once per shape with `torch.bmm`.
   ~177 params over maybe 10-20 distinct shapes => ~10x fewer matmul dispatches.
   **Expected ~80-100 ms/step = ~6-7%.** Train-time only, nothing ships to Rust.
   ⚠ Not bit-exact: batched bmm changes reduction order vs per-param mm. Muon is an optimizer, so
   this perturbs the trajectory like any nondeterminism — it needs an accuracy check, not just a
   speed one, and by the seed-pair doctrine a single run cannot resolve a <0.0005 difference.
2. **The 17,346 `cudaLaunchKernel`/step (199 ms).** The general elementwise mass — the real
   disease. CUDA graphs are the textbook cure and were shelved for variable shapes; `torch.compile`
   was shelved at an honest 1.05x, but that was measured **mixer-scoped at d=32**, before this
   dispatch count was known. Neither is a free retry (whole-graph compile hit Python 3.12's
   C-recursion cap in Dynamo), but the sizing argument for them is much stronger than it was.
3. **`indexing_backward_kernel`, 32 ms/step** — a surviving deterministic-indexing site. Find it
   (a third gather the PermGather work missed) and give it the same treatment. Bit-exact if done
   the way the previous two were.
4. **`aten::fill_` 7,227 calls / 34 ms** — a lot of zeroing for a 558k-param model; worth finding
   out what allocates-and-zeros that often.
5. **QAT-JIT — RUN 2026-07-27. Numerics settled; the SPEEDUP IS NOT REAL. ⚠ corrects the
   "~1.38x, worth ~1.5 days of the 10x run" claim in CLAUDE.md.**
   `scratchpad/qat_jit/` — 3 arms x 90 steps, warm-started from the iter-31 champion under the
   full q72u QAT env, with a NULL CONTROL (two runs at identical flags).

   | arm | steps/s |
   |---|---|
   | A nojit | 0.5901 |
   | A2 nojit (null control) | 0.6213 |
   | B jit | 0.6433 |

   * **NUMERICS: SETTLED.** Null control 0/160 mismatches (so the run IS reproducible and the
     comparison is valid) AND nojit-vs-jit 0/160 mismatches. **`RWKV_NO_JIT=1` is not required
     by QAT** — bit-identical over 80 real training steps on the CUDA `qat_lr_rank1` kernels,
     not merely the CPU reference the earlier smoke test covered.
   * **SPEED: NOT ESTABLISHED.** Two *identical-flag* runs differ by **5.3%**; jit vs the nojit
     mean is **1.06x**. The effect is inside the noise floor. The ~1.38x figure was measured on
     the NON-QAT body and does not transfer — consistent with the finding above, since
     TorchScript removes Python interpreter overhead but NOT the `cudaLaunchKernel` overhead this
     step is actually bound by. A real speed claim needs the 20-trial paired Wilcoxon protocol.
   * ⚠ **peak reserved 12.807 GB on a 12 GB card** in all three arms — the QAT config at
     MAX=32768 is over the ceiling and into WDDM paging, which may be depressing every arm. Worth
     re-measuring at a lower MAX before drawing conclusions about QAT speed at all.

## Method notes

- Bench: `scratchpad/profile_prep/run_bench_d80.cmd` (`RWKV_MAX_STEPS=90`, `RWKV_BENCH_WARMUP=40`).
- Profile: `scratchpad/profile_prep/run_profile_d80.cmd` (`RWKV_PROFILE_STEP=60`, count 10).
- Both use a scratch `SAVE_MODEL_PREFIX` so nothing real is touched; profile mode exits after the
  profiled steps and is off by default, so normal training stays byte-identical.
- ⚠ Read wall clock and kernel time TOGETHER. A CUDA-only profile showed 237 ms/step and looked
  like a fast step; it was 16% of the truth. `rwkv/profile_regions.py` exists to attribute the
  remainder to named code regions — PAVA and Muon are safe to annotate (`_pava_probe_loss` /
  `_pava_rectify_eval` are `@torch.jit.ignore`, Muon is never scripted), but **anything inside the
  scripted `SrsRWKV.forward` is not**: TorchScript barely supports `with`, and the project's
  history is that such a failure gets swallowed and turns the run HOLLOW rather than crashing.

## Round 1 of A/B results (2026-07-27 23:38) — everything inside the noise, and WHY

4-arm round-robin, 3 rounds, `scratchpad/profile_prep/run_speed_arms.cmd`:

| arm | n | median steps/s | within-arm spread |
|---|---|---|---|
| base (today's config) | 3 | 0.5639 | 3.5% |
| muon (`RWKV_MUON_BATCHED=1`) | 3 | 0.5848 | 5.5% |
| nojit | 3 | 0.5656 | 4.2% |
| compile (`RWKV_QAT_COMPILE=1`) | 3 | 0.5922 | 3.4% |

**Noise floor (widest within-arm spread) = 5.5%**, so: muon 1.037x, compile 1.050x (1.047x vs its
own nojit baseline), nojit 1.003x — **NONE established.** Each arm's flag was verified from its
banner, so these are genuinely different configurations, not a silently-ignored env var.

⚠ **The benchmark was measuring the wrong regime.** Every arm peaked at **12.83 GB reserved on a
12 GB card**, i.e. in WDDM paging. Two consequences: (a) paging variance is the likeliest source
of a 5.5% spread between *identical* configs — group order is NOT the culprit, `random.seed(12345)`
at `train_rwkv.py:33` makes `get_groups`' shuffle deterministic, so all arms see the same batch
sequence; (b) if paging dominates the step, removing 8,700 CPU dispatches cannot show up.

**=> `MAX_TRAIN_GLOBAL_LEN` is the suspect, and it was never swept for this trunk.** 32768 was
inherited from A18 so that step counts would pair with its trace; the only real sweep (to 110000)
was done at **d=32**, where peak was 9.44 GB. The model is now 2.5x wider. If 32768 is past the
VRAM cliff at d=80 then a SMALLER MAX is faster despite doing less work per step — the opposite of
the usual intuition, and a config change rather than a code change.

**Metric note:** compare MAX arms on **reviews/s**, never steps/s. Changing MAX changes both the
step count and the rows per step (this is exactly the trap that made iter 33's 16 h projection come
out at 31 h), so steps/s is not comparable across arms.

## ★ MAX sweep at d=80 (2026-07-27 23:52) — and the speed arms were INVALID

`scratchpad/profile_prep/run_max_sweep.cmd`, 2 rounds, density 0.08, batched Muon on.
**Metric is reviews/s** (steps/s is not comparable across MAX).

| MAX | groups (= steps/epoch) | steps/s | **reviews/s** | peak GB |
|---|---|---|---|---|
| 16384 | 43,354 | 0.68 | **4,807** | 8.798 |
| 24576 | 43,064 | 0.66 | **4,769** | 8.741 |
| 32768 | 22,346 | 0.59 | **8,868** | 8.860 |

**MAX=32768 is ~1.84x better than 16384 on throughput** — bigger batches win decisively, so the
inherited setting is right, and iter 33's forced MAX=16384 really does cost ~half the throughput
(that, not row count, is why its 16 h projection came out at 31 h).

### ⚠ THE SPEED ARMS ABOVE WERE MEASURED ON A CONTAMINATED GPU — treat them as void

Same MAX=32768 config, half an hour apart: **12.83 GB peak in the arms vs 8.86 GB in the sweep.**
The arms launched at 23:14:41, **one minute after** the 2.5 h rectified eval finished at 23:13, so
they inherited its GPU memory state and ran in WDDM paging; the sweep ran on a settled card. That
is the 5.5% noise floor, and it means "muon/compile/nojit all inside noise" is an artifact of the
measurement, not a finding about those arms. **Re-run them on a clean card before concluding
anything.**

**METHOD RULE (new): do not start a GPU benchmark immediately after a long GPU job.** Wait for
`nvidia-smi` memory.used to settle near idle (~1 GB here) and assert peak_reserved is in the
expected band; a benchmark whose peak is 45% above the same config's clean value is measuring
paging, not the change under test.

## ★★ THE BIG ONE: MAX=65536 is 1.61x the current throughput (2026-07-28 00:15)

Full sweep, clean card, 2 rounds each, density 0.08, batched Muon on:

| MAX | groups (steps/epoch) | **reviews/s** | peak GB | |
|---|---|---|---|---|
| 16384 | 43,354 | 4,807 | 8.798 | iter 33's forced value |
| 24576 | 43,064 | 4,769 | 8.741 | |
| 32768 | 22,346 | 8,607 | 8.860 | **current** |
| 40960 | 21,318 | 8,568 | 8.833 | |
| 49152 | 14,687 | 11,354 | 8.907 | |
| **65536** | **10,935** | **13,899** | **9.951** | **★ optimum** |
| 81920 | 8,696 | 13,477 | 12.382 | over the ceiling -> slower |

**1.61x vs the current MAX=32768**, for a config value. The curve reproduces the d=32 sweep's shape
exactly — throughput climbs to just under the VRAM ceiling then falls off a cliff (there: 110000 at
9.44 GB, 132k thrashing at -25%). This is literally the protocol's own step (methodology (f): "fix
the largest batch that ALMOST maxes the 12 GB VRAM"); it was simply never re-run after the trunk
moved from d=32 to d=80, because 32768 was inherited from A18 for step-pairing.

Within-arm spread on the clean card is 0.2-3.9%, vs the 5.5% seen on the contaminated one — so
these differences are far outside noise, unlike the four-arm result above.

### ⚠ IT IS NOT A FREE SPEEDUP — it halves the optimizer steps per epoch

Groups fall **22,346 -> 10,935 (-51%)**. Same data per epoch, half as many (twice as large) gradient
updates: the classic large-batch trade. Consequences before adopting:
- **LR and warmup must be re-tuned** — the protocol says batch size is structural and LR/warmup are
  tuned after it. `WARMUP_STEPS=200` is 0.9% of a 22,346-step epoch but 1.8% of a 10,935-step one.
- **It needs an ACCURACY check**, not just a stopwatch. A 1.61x that costs logloss is not a win, and
  by the research gate it would have to clear the champion in both modes.
- It re-bases step-pairing (vprune, champion traces). vprune is currently off, so no live blocker.

**Recommended next step:** one candidate run at MAX=65536 vs the iter-32 champion on the standard
gate. If accuracy holds, every subsequent experiment is 1.61x cheaper — which is exactly the
"speed up training so we can do more experiments" goal.
⚠ **iter 33 cannot use it**: its design needs `RWKV_PROBE_DENSITY=1.0`, which inflates rows ~2.54x,
so MAX=16384 is already near the VRAM envelope there. It stays at 16384.

## ★ THE RAM CLIMB IS OUR FETCH WORKERS (2026-07-28, after the hang)

Andrew asked whether the Reddit bot or `srs-benchmark/script.py` was "clogging up RAM". Measured
instead of argued — sampled every python/chrome working set 15 min apart:

| pid | start MB | end MB | delta MB | what |
|---|---|---|---|---|
| 26768 | 3341 | 3585 | **+244** | iter 33 fetch worker |
| 26988 | 3401 | 3640 | **+239** | iter 33 fetch worker |
| 25680 | 3358 | 3595 | **+237** | iter 33 fetch worker |
| 20100 | 3487 | 3720 | **+233** | iter 33 fetch worker |
| 28376 | 93 | 149 | +56 | chrome renderer |
| 18236 | 832 | 841 | +8 | **iter 33 MAIN process — flat** |

**4 workers x ~238 MB / 15 min = 3.8 GB/h**, matching the overnight climb (3.4 GB/h) almost exactly.
It is OUR run. The main process is flat, so it is not the model, optimizer or autograd — it is the
data path.

**Mechanism:** `prepare_batch.py:641` opens the LMDBs with default readahead on a **372 GB database
with 64 GB of RAM**. The OS reads ahead and those mapped file pages accumulate in each worker's
working set. The pages are clean and evictable, so this is not a leak in the classic sense — but it
is what drives "RAM used" into the 56-63 GB band where all three hangs occurred.

**⚠ `readahead=False` is NOT the fix on this platform.** The kwarg is accepted by py-lmdb 2.2.1, but
LMDB documents `MDB_NORDAHEAD` as **"not implemented on Windows"**, so it is very likely a no-op
here. Do not "fix" it that way and assume the problem is solved.

**PARTIAL FIX — `NUM_FETCH_PROCESSES` 4 -> 2, a toml change with no code edit.** Halves the number
of growing processes. It costs ~nothing because the same profiling run proved **fetching is not a
lever**: fetch waits are 2.3 ms of a ~1,450 ms step (0.1%), i.e. the loader is over-provisioned by
orders of magnitude.

**★★ BUT IT IS NOT SUFFICIENT AT MAX=65536 — MEASURED 2026-07-31 01:32, AND THE GUARD IS NOW
MANDATORY FOR ANY LONG UNATTENDED RUN.** During HP-tuner trial 2, with `NUM_FETCH_PROCESSES = 2`
already in effect, the two workers had reached **24.75 GB and 24.05 GB** after ~2.5 h of WS,
leaving **0.7 GB free of 63.9 GB**. That is not merely inside the 56-63 GB band that preceded all
three unexplained black-screen hangs — it is *deeper into it than any of them* (63.4 / 56.4 / 58.0
GB used). Halving the worker count did not halve the ceiling; it only halved how many processes
climb toward it.
**★ THE RATE, MEASURED PRECISELY 2026-07-31 from two consecutive guard firings** (05:03:55 free
14.0 -> 47.9 GB, then 05:58:06 free 13.9 -> 48.1 GB): **~34 GB re-accumulated in 54 minutes = ~38
GB/h**, with `NUM_FETCH_PROCESSES = 2`. That is **~10x the 3.8 GB/h** measured at MAX=16384 with 4
workers — each worker now holds the mmap pages for a 4x larger batch, and there is no sign of a
plateau below the point where the box runs out.
**Consequence, and it is why the guard is load-bearing rather than a nicety:** starting a phase
with ~50 GB free, the box re-enters the 56-63 GB hang band in **~1.2 h**. A WS phase is ~2.5 h, so
**every single WS phase would enter the band** without intervention — which fits the observed
history of unexplained hangs far better than any theory involving a rare trigger.
**What saves the untended case is that the climb RESETS at each phase boundary** (WS -> decay ->
eval kills and respawns the workers), so only phases longer than ~1.2 h are exposed. That is why
tuner trial 1 ran 4+ h without tripping the floor while trial 2 tripped it inside WS.
**One `EmptyWorkingSet` pass reclaimed 46.6 GB (1.0 -> 47.6 GB free) with zero effect on the run**
— steps kept advancing, fetch waits stayed at 0.004 s — confirming the pages are clean, file-backed
and cheap to drop, exactly as predicted.
**=> `scratchpad/run_ram_guard.cmd` (detached, `-FloorGB 14 -IntervalSec 60`) should be armed
alongside any multi-hour unattended training.** It was armed for the tuner at 01:37.

**Cleared as suspects:** the Reddit bot (`users_replied_to`/`ids_replied_to` are per-call locals, no
module-level growth, live footprint 0.01 GB) and — for THIS climb — `srs-benchmark/script.py`, whose
workers were near zero at measurement time. That script does nonetheless have a genuine accumulation
pattern worth fixing on its own: `script.py:639` submits every user at once and holds the entire
`futures` list, so each completed future retains its result (including the pre-serialized `raw`
JSON string) until the whole block exits. Read-only repo, reported not edited.

## ★★ BANKED: 1.155x from three changes (2026-07-29, clean card)

Re-ran the arms on a settled GPU after establishing the first attempt was void. **Noise floor
1.8%** (vs 5.5% contaminated), 3 rounds each, every arm's flag verified from its own banner:

| arm | median steps/s | spread | vs base | verdict |
|---|---|---|---|---|
| base | 0.5442 | 1.7% | — | |
| **muon** (`RWKV_MUON_BATCHED=1`) | 0.5731 | 1.4% | **1.053x** | REAL |
| nojit (`RWKV_NO_JIT=1`) | 0.5456 | 1.8% | 1.003x | not established |
| **compile** (`RWKV_QAT_COMPILE=1`) | 0.5728 | 0.0% | **1.053x** | REAL (1.050x vs its own nojit base) |
| **fetch2** (`NUM_FETCH_PROCESSES` 4->2) | 0.5773 | 0.7% | **1.061x** | REAL |

**COMBO (all three), 4 rounds: 0.6259 vs base 0.5418 = 1.155x REAL.** Predicted multiplicatively
1.177x, measured 1.155x — slightly sublinear, as expected when all three attack the same CPU time.

**Three things this settles:**
1. **Batched Muon is a genuine wall-clock win**, which the void run could not tell us — only that it
   cut matmul dispatches 35x.
2. **`torch.compile` deserved the retry.** It was shelved at 1.05x measured MIXER-SCOPED AT d=32,
   before the dispatch profile existed; it reproduces ~1.05x here but now with a mechanism that
   explains it. Note `nojit` alone is 1.003x, so compile's gain is NOT its `RWKV_NO_JIT` requirement.
3. **`fetch2` was queued as a RAM fix and is the biggest single win.** On a dispatch-bound step,
   two fewer worker processes mean less CPU contention with the main thread. It also halves the
   3.8 GB/h RAM climb implicated in the hangs — the rare change that is both.

### ⚠ Adoption is NOT uniform — one is free, two are not

- **`fetch2`: ADOPT FREELY — numerics-neutral, VERIFIED not assumed.** `DataFetcher.get(key)`
  (`data_fetcher.py:7-14`) blocks for the SPECIFIC group key and stashes out-of-order arrivals in
  `self.storage`, so the consumption order is fixed regardless of worker count. Fewer workers change
  throughput only.
- **Batched Muon + `torch.compile`: measured real, but BOTH perturb the trajectory** (batched `bmm`
  changes reduction order; inductor fusion changes it too). Adopting them silently would confound
  every future gate at 0.0001 sensitivity. **They need ONE accuracy run before adoption.** A short
  trace comparison cannot settle it — an optimizer perturbation compounds by design, so divergence
  is expected and says nothing about final quality; only a full run vs the champion does.

## ★★★ ADOPTED: 1.27x faster training at no accuracy cost (validated 2026-07-29 09:28)

The two trajectory-perturbing changes were validated by replicating the **iter-32 champion recipe
exactly** (full-run KD, same MAX, same data, dump step-aligned) with only the flags under test
changed, then gating rectified-vs-rectified against iter 32:

| metric | iter 32 | speedval | delta | p |
|---|---|---|---|---|
| ahead RECT | 0.300268 | 0.300204 | **+0.000064** | 0.012 |
| imm RECT | 0.267262 | 0.267309 | **-0.000047** | 0.9999 |

**PASS.** Both deltas round to 0.0000 at 4 dp and are **6-8x below the ~0.0004 cross-seed spread**,
with one mode marginally up and the other marginally down — the signature of an accuracy-NEUTRAL
change. (The `paired_pvalue` FAIL line is the gate for ACCEPTING AN IMPROVEMENT; it is the wrong
test for a speed change, where the question is whether there is systematic loss. There is not.)

### Real-world speedup is BIGGER than the microbenchmark said

| phase | iter 32 | speedval | ratio |
|---|---|---|---|
| WS (22,346 steps) | 4h23m (1.42 steps/s) | **3h27m (1.79 steps/s)** | **1.27x** |
| decay (5,586 steps) | 63 min (1.48 steps/s) | **50 min (1.86 steps/s)** | **1.26x** |
| train total | 5h27m | **4h17m** | **1.27x** |

The 70-step bench said 1.155x. The gap is fetch2 compounding: a short bench cannot see that 2
workers instead of 4 hold far less memory over hours. **Trust the long-run number.**

### THE STANDARD RUN ENV — put these in every new training `.cmd`

    set RWKV_MUON_BATCHED=1     REM batched Newton-Schulz (muon.py), 35x fewer matmul dispatches
    set RWKV_NO_JIT=1           REM required by torch.compile; worth ~0 on its own (1.003x)
    set RWKV_QAT_COMPILE=1      REM fuses the 26 mixer forwards

plus `NUM_FETCH_PROCESSES = 2` in the toml (**also halves the 3.8 GB/h RAM climb behind the
black-screen hangs** — the rare change that is both a speedup and a stability fix).

Defaults stay OFF, per the project norm that hooks are env-gated — so old runs stay reproducible
and these must be set explicitly.

⚠ **`--fetch-per-shard` in `eval_sharded.py` is still 4**, and the EVAL is the RAM-hungriest phase
(~26 GB/h measured, vs training's 3.8). The same lesson applies there and has not been applied yet.

## Items 4 and 5 CLOSED — stack attribution says they are not worth chasing (2026-07-29)

`RWKV_PROFILE_STACK=1` (new, opt-in; `with_stack` skews timings so it is for ATTRIBUTION runs only)
answered what grep could not:

| op | calls/step | self CPU | source |
|---|---|---|---|
| `aten::fill_` | 3,909 | 20.20 ms | **autograd internals — no frame in rwkv/** |
| `aten::fill_` | 511 | 2.31 ms | `rwkv_model.py:553` time_shift_gather |
| `aten::fill_` | 499 | 2.13 ms | `srs_model.py:899` _get_loss |
| `aten::fill_` | 429 | 1.96 ms | `srs_model.py:690` _get_loss |
| `aten::index` | 46 | 1.13 ms | `srs_model.py:690` _get_loss |
| `aten::index_select` | 43 | 2.17 ms | `rwkv_model.py:553` time_shift_gather |

**Verdict: CLOSED, both.**
- **`fill_` is ~68% autograd-internal gradient zeroing** — not ours to remove — and the whole op is
  ~29 ms of 915 ms CPU (**3%**).
- **The `indexing_backward` traffic comes from `time_shift_gather` and `perm_gather`, which ARE the
  2026-07-03 deterministic-indexing fixes.** What remains is the irreducible remainder of an
  optimization already applied, ~2% of wall clock.
- Both sit at or below the **1.8% noise floor**, so a fix could not be measured even if written.

This is the useful negative: the profile's ranked list looked like it had four items, and stack
attribution showed two of them were already solved.

## MAX=65536 — 1.32x more speed, but it COSTS ~0.0003 at unchanged LR (2026-07-30)

Champion trunk (iter-31 recipe, no KD) + the adopted combo, MAX 32768 -> 65536, gated
rectified-vs-rectified against iter 31:

| metric | iter 31 RECT | maxval | delta |
|---|---|---|---|
| ahead | 0.300802 | 0.301066 | **-0.000264** |
| imm | 0.267691 | 0.267997 | **-0.000307** |

**Read it as a small REAL loss, not noise.** Each delta alone is inside the ~0.0004 cross-seed
band, but BOTH modes moved the same direction. Contrast the speedval validation that PASSED:
+0.000064 / -0.000047, one up one down — that is what noise looks like. Two same-signed ~0.0003
deltas is the signature of a systematic effect near the resolution limit.

Mechanism is exactly as predicted: groups 22,346 -> 10,935, so an epoch gets **half the optimizer
steps** at the same LR and warmup. The large-batch literature's standard remedy is to scale LR
(linear 2x, or sqrt ~1.41x) — untested here.

**Speed it buys:** WS 3h27m -> 2h37m, decay 50 -> 40 min = **1.32x** on top of the adopted 1.27x,
i.e. **1.68x** vs the iter-32 baseline.

**DECISION IS ANDREW'S** (flagged when the sweep landed, not resolved unilaterally):
- **(a) REJECT** — keep the validated-neutral 1.27x. Safe; costs nothing.
- **(b) ACCEPT the loss** — 1.68x but every future champion carries ~-0.0003, which is 3x the
  accept threshold in the wrong direction. Not recommended in a phase where iterations win by
  +0.0005.
- **(c) RETUNE LR for the larger batch and re-test** — the principled fix, ~5.5 h for one run
  (2.7 h train at the new speed + 2.5 h eval). If it recovers the 0.0003, every subsequent
  iteration is 1.32x cheaper, which compounds over the ~50-iteration research plan.

### Ops note: the giant-user evals are VRAM-fragile, and it is environmental

Three eval attempts died with no traceback — users 5995 (266k reviews), 5905 (367k), 5002 (290k).
The flight recorder identifies it every time: **GPU OOM while the DESKTOP held several GB of VRAM**
(4.6 GB during the failures vs ~0.5 GB overnight, when these same users cleared three evals in a
row). The giant users need nearly the whole 12 GB card. The successful run started at 468 MiB used.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did NOT help and was dropped.
- **`scratchpad/maxval/run_maxval_eval3.cmd` is the resume runner: no `del`**, so `eval_sharded`
  skips completed users and a relaunch only re-risks the remainder. Use it for any big eval.

## ★★ WHAT MAX=65536 COST IS NOT AN LR PROBLEM — the `lr_mult` coordinate, 2026-07-31

MAX=65536 bought 1.61x but dropped the group count 22,346 -> 10,935, i.e. **half the optimizer
steps per epoch at unchanged LR**, and cost -0.000264 ahead / -0.000307 imm. The obvious
hypothesis was that the learning rate should scale with the batch, so the HP tuner's FIRST
coordinate was a joint `lr_mult` on **both** `PEAK_LR` (the 57,412 AdamW params) and
`RWKV_MUON_LR` (the 500,800 Muon params — tuning `peak_lr` alone would have moved ~10% of the
model). Four points, full runs, rectified eval on the 1000-user tune subset 5001-6000:

| lr_mult | peak_lr | muon_lr | ahead | imm | d_ahead | d_imm | objective |
|---|---|---|---|---|---|---|---|
| **1.00** | 0.00100 | 0.020 | 0.299250 | 0.266335 | — | — | **0.565585 (best)** |
| 1.41 (sqrt) | 0.00141 | 0.028 | 0.299547 | 0.266151 | -0.000297 | +0.000184 | 0.565698 |
| 2.00 (linear) | 0.00200 | 0.040 | 0.299749 | 0.266248 | -0.000499 | +0.000087 | 0.565997 |
| 2.80 | 0.00280 | 0.056 | 0.300272 | 0.266472 | -0.001022 | -0.000137 | 0.566744 |

**VERDICT: raising the LR does not recover the loss — it trades ahead for imm, and loses.**
- **ahead is strictly monotonic worse over all four points**, total swing **-0.001022**, ~2.5x the
  ~0.0004 seed-noise floor. A monotonic dose-response across four independent runs is far stronger
  evidence than any single delta, so this is a real effect, not a bad draw.
- **imm is an inverted U**: best at 1.41x (+0.000184 — only 60% of the +0.000309 needed) and
  NEGATIVE by 2.8x. So even the mode that MAX hurt most is not rescued by more LR.
- Objective is monotonic worse; baseline wins the coordinate outright.

**Consequences.**
1. **The standard batch-scaling heuristics (linear, sqrt) are actively harmful here.** Do not
   re-run this coordinate on a future MAX change; the answer is known and it is negative.
2. **The -0.0003 is looking like a genuine price for the 1.68x**, not a tuning oversight. The LR
   was the lever with a mechanistic reason to have moved; the survivors (warmup, muon/adamw RATIO,
   wd, clip, decay_ratio) all have weaker priors.
3. **`muon_lr_mult` remains a genuinely different question** and is still worth its trials: it
   changes the RATIO between the Muon and AdamW groups at fixed overall scale, whereas `lr_mult`
   moved both together. A wrong ratio would not have shown up in this coordinate at all.
4. If the remaining coordinates also come back empty, the honest framing for Andrew is a
   **Pareto choice** — 1.68x faster training for -0.0003 in both modes — not a bug to be fixed.
