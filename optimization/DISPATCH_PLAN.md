# The dispatch-bound speedup — plan

**Andrew, 2026-08-29:** *"Let's do it after the three arms but before continuing with new features
and algorithmic improvements."*

So the phase order is now: **three hybrid arms → THIS → features / algorithmic loop → 10x endgame.**

## Why this order is right, beyond scheduling

The profile below is measured at **d=80** (the iter-53 champion). If a hybrid arm promotes, the
trunk becomes d=32-shaped with 6-9 layer-steps instead of 13, and the dispatch profile changes
shape — fewer, smaller kernels, so *relatively more* dispatch-bound, not less. Profiling before the
arms report would characterise a trunk we may not keep.

## The target, and the honest ceiling

| per step (d=80, `TRAINING_SPEED.md`) | |
|---|---|
| wall clock | ~1,450 ms |
| GPU kernel time | **237 ms (~16%)** |
| self CPU time | 915 ms |
| op dispatches | **90,576** |
| `cudaLaunchKernel` | 17,346 calls, **199 ms** |
| `cuLaunchKernel` | 3,499 calls, 48 ms |
| fetch wait | 2.3 ms — not a lever, re-confirmed live |

Confirmed live on hybrid arm A, 2026-08-29 (60 samples / 30 s): **mean GPU utilisation 31%, 72% of
samples below 50%, 17% at literal 0%.** The card is idle most of the time.

**Amdahl ceiling = 1450/237 ≈ 6x** with dispatch made free. **Realistic target 2-3x.** Nobody
reaches the ceiling, and saying 6x out loud invites measuring against a number that was never
achievable.

⚠ **The "9x from a smaller model" framing is wrong and should not be repeated.** 9x is an
*arithmetic* ratio and arithmetic is 16% of the step. Arm A's observed 1.41x over the champion
comes mostly from having 9 layer-steps instead of 13 — i.e. **fewer dispatches** — not from the
width cut.

## ★ THE METHODOLOGICAL CONSTRAINT THAT SHAPES THE WHOLE PLAN: BIT-EXACT OR RE-BASE

Every speedup banked so far was **bit-exact** — the allocator scratch, deterministic indexing
(1.5x), the QAT kernels (37x). That was not an aesthetic choice. A training-time change that
alters numerics moves the optimisation trajectory, so **every future candidate would be measured
against a champion trained on a different stack**. Re-basing means re-running the champion: ~6.6 h
train + ~2.9 h eval ≈ **9.5 h**, and by the seed-pair doctrine a single re-run cannot even resolve
a <0.0005 difference.

So the work splits in two, and the first tier is worth far more per hour than its raw speedup
suggests:

**TIER 1 — bit-exact, no re-base, bankable immediately.**
**TIER 2 — changes numerics, costs a 9.5 h re-base plus a seed pair before anything else can be
measured.** A tier-2 win must be large enough to be worth that, which means roughly ">1.5x" rather
than ">5%".

## Ranked approaches

### 1. CUDA graphs with static shape buckets — TIER 1 if done right, and the sizing argument INVERTED

Graphs replay a captured sequence with ~one launch, which is the textbook cure for a launch-bound
step. It was shelved as *"variable shapes, ~1.1-1.3x only"* — **both halves of that are now wrong.**

* The 1.1-1.3x estimate predates knowing the step is 85% dispatch.
* **The blocker was static shapes, and padding to fixed buckets is now CHEAP.** Padding wastes GPU
  *compute*, which is 16% of the step. Even 2x padding waste costs ~3% of wall clock, against
  collapsing ~247 ms of launch overhead. **The thing that made graphs unattractive is exactly the
  thing that stopped mattering.**

Bit-exact **if** padded rows are masked out and no reduction crosses a padded element. That is a
real condition to verify, not to assume — the WKV recurrence and the per-split gathers both need
checking. Verify with the existing golden-tensor method (`scratchpad/qat_speed/golden_gen.py`),
which is how the QAT kernel rewrite was proven bit-exact.

Real obstacles, in the order they will bite: no CPU syncs inside the captured region (the NaN-check
`except` is one), fixed memory addresses, and the per-split dynamic shapes that motivate the
bucketing in the first place.

### 2. The two known dispatch sites — TIER 1, small but cheap and certain

* `indexing_backward_kernel`, **32 ms/step** — a third deterministic-indexing gather that the
  PermGather work missed. Bit-exact if given the same treatment as the previous two.
* `aten::fill_`, **7,227 calls / 34 ms** — a lot of zeroing for a 558k-param model. Find what
  allocates-and-zeros that often.

Together ~66 ms of ~1,450 = ~4.5%. Not exciting alone, but bit-exact, well-understood, and the
`fill_` count is a symptom worth understanding regardless of the fix.

### 3. `torch.compile` whole-graph — TIER 2

Shelved at "an honest 1.05x", but that was measured **mixer-scoped**: it fused inside the mixers and
never attacked the cross-op dispatch count, which is where the 199 ms lives. Whole-graph hit Python
3.12's fixed C-recursion cap in Dynamo — raisable via a larger thread stack, or 3.13 — but the
failure mode there was RecursionErrors **silently eaten by the NaN-safety `except`, producing hollow
steps and a fake 1.27x profile**. Any retry must count "Exception caught" before trusting a number.

`mode="reduce-overhead"` is CUDA graphs underneath, so approach 1 and this one converge; the
difference is whether we own the capture or Inductor does.

## Measurement protocol — the rules already paid for

* **reviews/s, never steps/s**, whenever `MAX_TRAIN_GLOBAL_LEN` differs between arms.
* **Do not start a GPU benchmark immediately after a long GPU job** — it voided a whole round of
  A/B results on 2026-07-27. Wait for the card to settle.
* **Within-arm noise floor is 5.5%**, so a 1.05x claim is not a claim. Speed accepts use the
  20-trial paired Wilcoxon at p<0.01 (`optimization/wilcoxon_speed.py`, `--trials 20`).
* **Include a null control** — two arms at identical flags. The QAT-JIT measurement found identical
  configs differing by 5.3%, which is what turned a "1.38x" into "not established".
* **Check VRAM headroom.** The QAT-JIT arms sat at 12.807 GB on a 12 GB card and every arm may have
  been depressed by WDDM paging. Any measurement above ~11 GB is suspect.

## First actions when the GPU frees

1. Re-profile the **winning** trunk with `RWKV_PROFILE_STEP` — the numbers above are d=80 and may
   not be the model we keep.
2. Confirm the dispatch split still dominates at that shape, and re-derive the Amdahl ceiling.
3. Then tier 1, in the order above. Do not start tier 2 until tier 1 is banked, because a tier-2
   re-base makes every tier-1 measurement harder to attribute.

---

## MEASURED 2026-08-30: the MAX sweep, and it refutes the hypothesis it was built on

The champion bench showed `peak_reserved_gb=13.069` on a 12.28 GB card, so every timing this
project owns was taken while the driver was paging. The obvious lever was to shrink
`MAX_TRAIN_GLOBAL_LEN` until the step fit. Measured on a quiet machine, KD off (forced -- see
below), 120 warmup + 100 timed steps per arm, **65536 run first AND last as a drift control**:

| MAX | reviews/s | peak reserved GB |
|---|---|---|
| **65536** | **22,985.7** / **23,386.7** | 13.072 |
| 49152 | 20,410.1 | 13.032 |
| 40960 | 16,967.8 | 12.994 |
| 32768 | 16,474.9 | 13.034 |
| 24576 | 12,356.4 | 12.925 |

The two control arms differ by **1.7%**, so every gap above is real.

**A 2.67x reduction in MAX costs 47% of throughput and buys 1.1% of VRAM.** Peak reserved memory is
therefore **essentially independent of MAX** -- the 13 GB is not the batch. 65536 is already the
best of the five, so there is nothing to win here and the lever is closed.

**★ THE USEFUL PART IS THE INVERTED QUESTION.** If VRAM does not scale with MAX, the reason to stay
at 65536 was never memory, and MAX **above** 65536 becomes worth testing -- it might buy throughput
at the same 13 GB. That is the next cheap experiment (~10 min), not another reduction.
⚠ It is not free accuracy-wise: MAX sets the group count and therefore the optimizer steps per
epoch, which is why iter 34's move to 65536 cost 0.0003 at the old LR. A larger MAX would need the
LR retuned with it -- which phase 5 is doing anyway.

**And the VRAM question needs a different instrument.** Whatever holds 13.0 GB is MAX-invariant, so
it is model/optimizer state, allocator behaviour or fragmentation -- not the batch. Note
`torch.cuda.max_memory_reserved` is a high-water mark that `empty_cache` does not reset, so the
first thing to establish is whether 13 GB is ever *resident* or merely *reserved once*.

### ⚠ A KD RUN CAN ONLY BE BENCHED AT ITS OWN MAX

Every non-baseline arm of the first attempt died with exit 43 and
`labels_sum 40756926045.9 vs dumped 124347298553.5, batch stream diverged`. The KD dump stores
teacher logits keyed to the batch stream, and **MAX *is* the batch stream**. That is the alignment
guard working exactly as designed, and it means a sweep is necessarily a PLAIN-recipe measurement.
KD costs a roughly per-REVIEW amount, so it dilutes the spread slightly without reordering it --
compare arms to each other, never to the 19,528.6 rev/s KD figure on record.

### ⚠ A DISPATCH-BOUND BENCHMARK NEEDS A QUIET CPU

The first attempt's own 65536 arm read **16,191 rev/s against 19,529 on record at the same MAX** --
a 17% deficit -- because a 6-thread CPU rebuild was running. The step is ~85% CPU-bound, so a CPU
co-tenant hits it directly. The standing rule "no co-tenant GPU work during gate-critical runs"
extends to **CPU** for anything dispatch-bound.

---

## ★★ CORRECTION 2026-08-31: THE CARD IS NOT FULL, AND THE PAGING PREMISE WAS WRONG

Phase 1 opened on `peak_reserved_gb=13.069` against a 12.28 GB card, read as "every timing this
project owns was taken while the driver was paging". Sampled during a REAL WS phase of the e2s
re-base -- 171 samples over 75 s, `nvidia-smi` (driver-side, which is what actually pages):

| | MiB |
|---|---|
| minimum | 902 |
| median | 7,486 |
| **maximum** | **8,182** |
| card | 12,282 |

**Steady-state training peaks at 8.2 GB and leaves ~4 GB of headroom.** Utilisation median 75%,
minimum 3% -- the dispatch-bound picture is intact, the memory picture was not.

**Why the two numbers disagree, and which one to trust.** `peak_reserved_gb` is
`torch.cuda.max_memory_reserved()`, a HIGH-WATER MARK that `empty_cache()` does not reset. With
`RWKV_EMPTY_CACHE_EVERY=1` the allocator hands cached blocks back to the driver every step, so a
brief 13 GB reservation is released and never appears again -- while the high-water mark keeps
reporting it forever. `nvidia-smi` reports what the driver has actually handed out, so it is the
number that governs paging. A transient over-subscription did happen once; it is not the regime
training runs in.

**Three consequences:**
1. **The MAX sweep's null now has a mechanism.** VRAM did not move with MAX because VRAM was never
   the binding constraint. That result is unchanged and still closes downward MAX.
2. **MAX ABOVE 65536 is the live experiment**, and the headroom says it can fit. ~10 min to test.
3. **Any future VRAM claim must come from `nvidia-smi` sampling, not from `max_memory_reserved`.**
   Report the max of a sampled series; a high-water counter cannot distinguish "resident" from
   "touched once during warmup", and those imply opposite actions.

⚠ Generalisable, and this is the third instance in two weeks: **a high-water statistic cannot
describe a regime.** Iter 51 fitted on a median and missed a 1.76e7 blow-up; the `a`-is-dead probe
used a resting value where a bound was needed; here a peak stood in for a steady state. Match the
statistic to the question -- max for safety bounds, median/distribution for regimes -- and say
which one is being reported.

---

## ★★ THE UPWARD MAX SWEEP (2026-08-31): +14% REVIEWS/S, BUT NOT A FREE SPEEDUP

Run on the e2s train db (SSD-backed), plain recipe, KD forced off, 120 warmup + 100 timed steps
per arm, **65536 first AND last as the drift control**.

| MAX | reviews/s | groups (optimizer steps/epoch) | peak_reserved GB |
|---|---|---|---|
| 65536 | **24,119** / **23,442** | 10,935 | 11.788 |
| 81920 | 26,125 | 8,696 | 11.729 |
| **98304** | **27,178** | **7,208** | 11.727 |
| 114688 | 21,451 | 6,150 | **12.965** |

The two control arms differ by **2.9%**, so every gap above is real. 98304 beats the 65536 mean by
**+14.3%**.

**The cliff at 114688 has a clean mechanism and is visible in the data.** `peak_reserved` is flat at
11.73-11.79 GB from 65536 through 98304 and then jumps to **12.965 GB -- past the 12,282 MiB card**.
Throughput collapses 21% at exactly that point. That is what paging actually looks like, and it is
the first time this project has caught it happening rather than inferring it.

**It also explains the earlier "VRAM is MAX-independent" finding rather than contradicting it:** the
allocator reuses its existing blocks across a wide range of MAX, so reserved memory stays flat until
the working set genuinely outgrows them, then steps. Flat-then-cliff, not proportional.

### ⚠ THIS IS A TRADE, NOT A WIN, AND THE GROUP COLUMN IS WHY

An epoch is a fixed number of reviews, so 98304 finishes one in **12.7% less wall clock**. But it
does so in **7,208 optimizer steps instead of 10,935 -- 34% fewer updates**. That is precisely the
iter-34 situation: halving the group count cost **0.0003 in both modes at the unchanged LR**, and
was only recovered by retuning.

**So MAX=98304 is a phase-5 item, not something to adopt now.** Phase 5 (HP tuning WITH QAT) retunes
the LR anyway, and MAX is structural, so it belongs in that sweep as a lever rather than being fixed
beforehand. Adopting it today would also re-base the champion a second time inside a week.

**What is banked regardless:** the downward direction is closed, the upward optimum is located, the
cliff is located and explained, and the accuracy cost of moving is a known quantity rather than a
guess.

---

## ★★★ ROUND 1, MEASURED CLEAN (2026-08-31): THE STEP IS GPU-BOUND, AND CUDA GRAPHS ARE THE WRONG TARGET

Andrew flagged that another Claude was loading the CPU when the 2026-08-30 profile was taken. That
splits the evidence rather than voiding it: **CPU self-times are host-side waits and would inflate;
GPU kernel time is measured on-device and would not.** Re-taken on a genuinely quiet machine
(GPU free, CPU 0%):

| | contended (08-30) | **clean (08-31)** |
|---|---|---|
| total GPU kernel time | 1,416.74 ms/step | **1,368.84 ms/step** |

**3.4% apart -- so the device-side figure was robust, and `DISPATCH_PLAN`'s headline of 237 ms/step
is wrong by 5.8x.** It is not contention and it is not noise. The step is **GPU-BOUND**.

**=> CUDA GRAPHS ARE DEMOTED FROM "TOP CANDIDATE" TO "ADDRESSES A NON-BOTTLENECK".** They remove
launch overhead. Launch overhead is not what this step is spending its time on. The whole
"Amdahl ceiling ~6x, realistic target 2-3x" framing was derived from the 237 ms figure and does not
survive it. Nothing structural should be built on that number again.

### Where the time actually goes

| kernel | share | ms/step |
|---|---|---|
| `aten::_index_put_impl_` | 18.95% | ~259 |
| `indexing_backward_kernel` | 18.67% | ~256 |
| **indexing, combined** | **37.6%** | **~515** |
| wkv plain recurrence | 10.6% | 145 |
| gemm (all linear layers) | 3.2% | 43 |

**Indexing costs more than the WKV recurrence and every matmul combined.** That is the target.
Round 2 sizes how much of it is the determinism tax (`RWKV_DETERMINISTIC=1` forces sort-based
scatter instead of atomics) versus intrinsic to the interleaved schedule, which scatters and
gathers per stream per round across 13 layer-steps.

### ⚠⚠ THE ALLOCATOR LEAD WAS MINE AND IT WAS BACKWARDS -- THE EXISTING RULE IS CONFIRMED

Hypothesis: `RWKV_EMPTY_CACHE_EVERY=1` costs ~920 ms/step in `cudaFree`+`cudaMalloc`, and the rule
justifying it ("allocator creep -> WDDM paging -> 4x slowdown") rests on a premise I had just
falsified -- real training peaks at 8.2 GB on a 12.28 GB card. Measured, arms alternating 1/0/1/0:

| `empty_cache` | reviews/s | peak_reserved |
|---|---|---|
| **1 (current)** | **24,152.9 / 23,928.0** | 11.788 |
| 0 | 17,821.6 / 17,873.2 | 11.914 |

**every=0 is 26% SLOWER**, and the profile says why: GPU kernel time goes **1,369 -> 4,321 ms/step**,
a 3.2x blowup matching the "4x slowdown" the rule predicted.

**THE ERROR IS WORTH MORE THAN THE EXPERIMENT: I MEASURED THE TREATED STATE AND CONCLUDED THE
TREATMENT WAS UNNECESSARY.** The 8.2 GB peak was observed *with the flag active* -- it is the flag
doing its job, not evidence the flag is redundant. Same shape as reading a low fever as proof the
antipyretic was pointless.
**The general form: before removing a control, ask whether the evidence against it was produced by
it.** Cheap to check here (35 min) and it converted an unexamined rule into a measured one.
`RWKV_EMPTY_CACHE_EVERY=1` STAYS, now on direct evidence.

---

## ★★★ BANKED 2026-09-01: PermGather on the INTERLEAVED path -- +5.5% throughput, BIT-EXACT

`perm_gather` was wired on the SEQUENTIAL stream gather (`srs_model.py:1080`) and **missed on the
INTERLEAVED one** (`:1249`) -- which has been the champion's path since iter 41 and runs that gather
once per layer-step per split, 13 layer-steps deep. Its stock `index_select` backward is the
deterministic sort-based `index_add` that `_PermGather`'s own docstring prices at *"~43% of the
whole training step"*.

Three independent numbers agreed before the fix was written: the docstring's ~43%, the measured
indexing share (37.6% of GPU time), and `RWKV_DETERMINISTIC=0` being worth +30.9% throughput.

| | reviews/s | GPU kernel ms/step |
|---|---|---|
| `RWKV_PERM_GATHER=0` (stock) | 23,474.9 / 23,677.4 | 1,213.1 |
| **`=1` (fixed, default)** | **25,177.2 / 24,561.9** | **892.5** |

**+5.5% throughput; 320.6 ms/step of GPU kernel time removed (-26.4%).**

### It is BIT-IDENTICAL, and that was verified before it was used

`_PermGather.forward` is `clamp(idx, min=0)` then `index_select` -- character-identical to what
`:1249` already did -- and the backward differs only by a row-0 pad-sum that adds exact zeros. But
"should be identical" is not "is", and it mattered: the fixc arm would run WITH the fix while the
e2sc re-base ran WITHOUT, so any trajectory perturbation would confound the interval measurement.
40 steps, both arms, compared line-for-line at full printed precision: **BIT-IDENTICAL**. Tool:
`scratchpad/dispatch/cmp_traces.py`, which returns a DISTINCT exit code when a trace has no loss
lines, so a test that never ran cannot read as a pass.

**=> No re-base. No seed pair. The champion's numbers stand unchanged.** This is the fourth
bit-exact speedup banked (allocator scratch, deterministic indexing, QAT kernels, this).

### ⚠ THE THROUGHPUT AND GPU-TIME NUMBERS DISAGREE, AND THE GAP IS THE NEXT LEAD

26.4% of GPU kernel time removed bought only 5.5% of throughput. That is not a contradiction -- it
means the step is **no longer GPU-bound after the fix**. Removing GPU work now exposes CPU/dispatch
overhead underneath.

**So the "dispatch-bound" thesis this plan opened with is partly rehabilitated -- but only AFTER
this fix, and never at the 237 ms/step it claimed.** The honest sequence is: the step WAS GPU-bound
(1,213 ms/step of kernel time), the dominant cost was a missing optimisation rather than anything
structural, and removing it moves the bottleneck. CUDA graphs are worth re-examining now, against a
freshly measured profile -- not against the stale figure.

### What is now closed

* MAX downward -- 65536 already optimal.
* `empty_cache=0` -- 26% SLOWER; the existing rule is confirmed by measurement.
* CUDA graphs *as justified by the 237 ms figure* -- that number predates interleaving (the
  comment in `train_rwkv.py` is dated 2026-07-27; interleaving landed 2026-08-11).
* Interleaving costs 372 ms/step of GPU time, but explains only ~38% of the gap to that figure --
  the trunk sits at 841 ms/step even with it off, so the stale number is not fully explained by it.

---

## ROUND 5 PLAN (2026-09-01): where the time is NOW, after the PermGather fix

Post-fix profile, quiet machine. **The bottleneck has moved, so the old ranking is void.**

| GPU (892.5 ms/step total) | share |
|---|---|
| `rwkv7_wkv_backward_bfloat16` | 11.9% |
| `triton_red_fused_mul_native_group_norm...` | 8.6% |
| `rwkv7_wkv_backward_time_parallel_final` | 7.6% |
| indexing (was 37.6%) | **9.4%** |

Nothing dominates any more -- the GPU profile is flat, and 80% of it is the "other
(elementwise/reduce/copy/optim)" bucket spread over many small kernels.

| CPU (1,064.5 ms/step self) | ms/step | calls/step |
|---|---|---|
| `cudaFree` | 184.5 | 391.6 |
| `cudaLaunchKernel` | 168.3 | **15,243** |
| `cudaStreamSynchronize` | 121.5 | **273** |
| `cuLaunchKernel` | 49.7 | 3,513 |

⚠ **CONTENTION CONFIRMED AS ANDREW SUSPECTED: total self CPU was 2,399.7 ms/step in the 08-30
profile and is 1,064.5 here -- inflated 2.25x by the other Claude's load.** The GPU figure moved
only 3.4% over the same pair. Host-side waits inflate; device time does not. Any future CPU-side
number must come from a quiet machine, and it is worth checking before profiling.

### The two candidates, ranked

**1. CUDA graphs -- NOW justified, on a fresh measurement rather than the stale one.** 18,756
launches/step costing 218 ms of a 1,121 ms wall step (~19%). This is the first time the evidence
actually supports them: before the PermGather fix the step was GPU-bound, and the 237 ms/step
figure that originally motivated them predates interleaving entirely.

**2. `cudaStreamSynchronize` -- 273 calls/step, 121 ms.** Partly located by reading:
`srs_model.py:1604-1611` makes FOUR `.item()` calls INSIDE the forward (`ahead_n`,
`ahead_equalize_n`, `imm_n`, `imm_binary_equalize_n`), and `train_rwkv.py:1363` four more for the
per-step print. **The cost of a mid-forward sync is not its own ~0.4 ms -- it drains the pipeline,
because the CPU stops queueing work until the GPU catches up.** These look like reporting counts,
so they should be deferrable: keep them as tensors and resolve once at the print. 273 is far more
than the 8 found, so the rest still needs locating -- do not assume these are all of it.

⚠ **DO NOT EDIT `srs_model.py` WHILE AN ARM IS RUNNING.** A chain's later phases are new processes
that import whatever is on disk then; editing mid-chain silently changes the next phase. This was
violated on 2026-09-01 (the PermGather edit landed while the fixc arm was in its dump) and cost
12 minutes plus a restart. The dump was unaffected only by luck -- it uses the d=128 teacher arch.

### Closed, with the evidence

* MAX downward (65536 optimal) · `empty_cache=0` (26% SLOWER, rule confirmed) · the 237 ms/step
  premise (stale, predates interleaving) · interleave cost (372 ms/step, but explains only ~38%
  of the gap to that figure).

### ROUND 5, RESOLVED ON EXISTING DATA (2026-09-01) -- no GPU needed

**Lead 2 (the 273 stream syncs) is CLOSED: they are CAUSED BY `empty_cache`, not by our code.**
Comparing the two round-1 profiles settles it -- with `empty_cache=0`, `cudaStreamSynchronize`
disappears from the CPU table entirely.
⚠ Precisely: it is absent from the whole 32-row dump, having been the **3rd largest entry at
121.5 ms** with the flag ON. The dump is TRUNCATED, so this is "fell by at least an order of
magnitude", not a literal zero -- which is all the argument needs, and is what the evidence
actually supports. (Checked because the first version of this note said "vanish" on the strength
of a `head -10`, which would not have distinguished absent from merely-demoted.)

| | `empty_cache=1` | `empty_cache=0` |
|---|---|---|
| `cudaFree` | 391.6 calls, **184.5 ms** | 80.2 calls, **1,198.1 ms** (15 ms EACH) |
| `cudaStreamSynchronize` | 273 calls, 121.5 ms | absent |
| `Command Buffer Full` | -- | 45.3 ms (WDDM pressure) |

`empty_cache=1` spends ~306 ms/step on cheap frees and syncs to avoid ~1,240 ms of catastrophic
frees plus command-buffer stalls. That is the ~900 ms/step difference, and it reconciles exactly
with the measured 26% throughput gap. **The syncs are the price of a mechanism that repays it four
times over -- removing them means removing `empty_cache`, which is strictly worse.**

⚠ The eight `.item()` calls found by reading (`srs_model.py:1604-1611`, `train_rwkv.py:1363`) are
real and still worth deferring on principle -- a mid-forward sync drains the pipeline -- but they
are 8 of 273, i.e. NOT the story. Do not spend a round on them expecting the 121 ms.

### ⚠⚠ AND THIS PUTS CUDA GRAPHS IN DIRECT CONFLICT WITH A CONFIRMED WIN

Graph capture requires **stable memory addresses**; `empty_cache()` every step is the exact
opposite, and it is worth 26% by measurement. PyTorch's graph pool is separate from the general
caching allocator, so the two MAY coexist -- but that is a hypothesis, not a fact, and it is the
first thing to test rather than the last.

**So the round-5 order is: (a) verify a captured graph survives per-step `empty_cache` at all,
(b) only then measure the launch-overhead win.** If they cannot coexist, the choice is 19% of
launch overhead against a measured 26%, and graphs lose. That inverts the plan's original ranking
for the second time, and on the same principle each time: the number that justified them was never
measured against the configuration we actually run.
