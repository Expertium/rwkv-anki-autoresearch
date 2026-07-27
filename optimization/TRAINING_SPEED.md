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
