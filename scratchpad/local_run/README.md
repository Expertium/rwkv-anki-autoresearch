# Linux local-training reproduction + a GPU-allocator finding

Scripts to run the current champion recipe (iter 41: interleave + fine-to-coarse order) on a
single Linux box, plus a training-pipeline speed/stability finding turned up while doing it.
Xemor's own machine, not Andrew's official Windows research box — nothing here touches
`CLAUDE.md`, `research_log.jsonl`, or any promoted-champion artifact.

## What's here

- `local_ws.toml` -- clone of `scratchpad/iter41_ilv/i41_ws.toml` with Linux-local paths
  (`VALIDATE_DATASET_LMDB_PATH` etc. instead of the Windows `F:/rwkv_lmdb/...` paths).
- `run_ws.sh` -- bash port of `scratchpad/iter41_ilv/run_iter41.cmd`'s WS phase. Same env as
  the champion recipe **minus `RWKV_KD_MIX`/`RWKV_KD_ALPHA`** -- the distillation teacher dump
  (`C:\rwkv_kd_dump\t128_seedpair_65k`, ~7 GB) is a Windows-machine artifact not present here,
  so this reproduction trains without KD. Not a like-for-like number vs. the official iter 41
  result for that reason; useful for pipeline/speed work, not for logloss comparisons.
- `run_bench.sh` / `run_profile.sh` / `run_profile2.sh` -- speed-experiment harnesses using
  `train_rwkv.py`'s existing `RWKV_MAX_STEPS`/`RWKV_BENCH_WARMUP` (steady-state steps/s + peak
  VRAM) and `RWKV_PROFILE_STEP`/`RWKV_PROFILE_COUNT` (bucketed CUDA-kernel + CPU self-time
  profile) hooks -- no new instrumentation, these already existed in `train_rwkv.py`.

## Finding: `RWKV_EMPTY_CACHE_EVERY=1` (whole-run) costs ~500 ms/step here, and isn't even safe

The champion recipe (iter 41, and everything since iter 34) trains with
`RWKV_EMPTY_CACHE_EVERY=1 RWKV_EMPTY_CACHE_WINDOW=0` -- clear the CUDA allocator's cache on
*every* step, for the *whole* run, as a fragmentation guard. On this machine (RTX 4070 Ti,
12 GB, shared with a desktop compositor/remote-desktop session eating ~0.6-3 GB) that guard
still wasn't enough -- the very first bench run reproducing the exact champion env hit a real
`CUDA out of memory` mid-run.

`RWKV_PROFILE_STEP` pointed straight at why: of ~1.2 s CPU wall-clock per step, **433 ms was
`cudaFree` and 73 ms was `cudaMalloc`** (414 and 413 calls/step respectively) -- almost all of
it attributable to the per-step `empty_cache()` forcing the allocator to release and immediately
reacquire memory. GPU kernel time was ~1.09 s/step in the same profile, so the process was
CPU/allocator-bound, not GPU-compute-bound -- consistent with the low, spiky `nvidia-smi`
utilization (SM% swinging roughly 3-45%, power 44-100 W of a 285 W card) that prompted this dig.

Switching to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the exact remedy PyTorch's own
OOM message suggests) and turning `RWKV_EMPTY_CACHE_EVERY` **off** together:
- fixed the OOM (300-step steady-state run: 0 crashes, peak reserved 8.28 GB, stable -- vs. the
  official env's OOM under the same architecture/batch config on this GPU),
- cut CPU wall-clock to ~685 ms/step (`cudaFree` down from 414 calls/step to ~95, mostly
  residual allocator housekeeping under `expandable_segments`, not the full clear),
- and roughly **2.6x'd steady-state throughput** in the 300-step benchmark (25,019 reviews/s
  vs. 9,698 reviews/s for the `expandable_segments`-but-cache-still-on control -- the official
  env's own number isn't usable as a baseline since it crashed).

`RWKV_QAT_COMPILE=1` (torch.compile on the mixer forwards) turned out to be load-bearing for
this, not just a speed nicety: turning it off under the same allocator settings caused *worse*
OOM-retry thrashing (repeated caught-and-retried `torch.OutOfMemoryError`, 0.26 steps/s) --
compiling fuses the elementwise chains and measurably lowers peak memory, so it's part of why
the fix works, not orthogonal to it.

**Caveats -- this is NOT the project's own Wilcoxon speed protocol (§11, 20 paired trials,
p<0.01):** the numbers above are single/few-run benchmarks on one machine, meant to surface and
motivate the finding, not to promote it into the champion recipe unilaterally. Two things worth
someone (Andrew's side) checking before adopting this on the Windows research box: (1) whether
the *same* OOM/fragmentation pressure exists there -- it may not, if that GPU is dedicated to
training with no desktop compositor sharing it, in which case the win is smaller (`empty_cache`
overhead should still be real, just maybe not the "prevents an actual crash" part); (2) whether
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` behaves identically on the Windows CUDA driver.

Separately (not a speed lever, an ops bug): a background `data_processing` build left orphaned
worker/writer processes running well after its parent script reported completion, holding CPU
and (transiently) GPU memory into later experiments and causing avoidable OOM cascades in a
few of the runs during this investigation -- same failure mode `CLAUDE.md` already flags for the
eval path ("CHECK + KILL ORPHAN PYTHONS after every run"); apparently not unique to eval.

## What's still open (a real GPU-kernel-time item, not a pipeline knob)

Even after the fix above, the GPU kernel-time profile is dominated (~40%) by
`aten::_index_put_impl_` + its backward `indexing_backward_kernel` -- almost certainly the
`RWKV_INTERLEAVE` round-robin schedule's per-round gather/scatter (new in iter 41, after the
project's earlier `PermGather`/flat-row gather optimizations, which predate interleaving and
don't cover it). A permutation-based rewrite of that op (same result, same trick as the
existing `PermGather`) is a plausible further win, but it's a real code change to
`rwkv_model.py` needing the project's three-way parity re-verification -- out of scope here.
