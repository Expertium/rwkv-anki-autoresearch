# RWKV 5k/5k research log — scaling to 5000-user training

> New phase (2026-06-30, Andrew). The 100/100 and 1500-user workbenches found the H=2/K=16 champion;
> this phase trains it at FULL 5000-user scale and measures it against the original big model on a
> proper held-out half. Canonical numeric log for this phase — append every run here. The pre-5k
> history lives in `research_log.md` / `log.md` / `HISTORY.md`.

## The setup
- **Train:** users **1–5000**.   **Eval:** users **5001–10000** (disjoint held-out half).
- **Compute budget (the run is slow at 5k):** **2 WS epochs + 0.5 decay epochs** (cosine).
- **Our model:** the H=2/K=16 champion (d=32, 2 heads × K=16, layers [1,4,3,3,3], 193,724 params,
  per-card WKV state = two 16×16 per-head matrices). Recipe env: `RWKV_N_HEADS=2 RWKV_HEAD_DIM=16`,
  `RWKV_EMPTY_CACHE_EVERY=0`, `RWKV_DETERMINISTIC=1`, `RWKV_AUGMENT_SEED=1234`, HP from the tuner.

## ★ BASELINE TO BEAT — the big old model on 5001–10000
The original leaderboard d=128 model **`pretrain/RWKV_trained_on_101_4999.pth`** (2.76M params,
4 heads × K=32), evaluated on **5001–10000** — a genuine held-out set for it (it was trained on
101–4999). Eval via the arch-swap `scratchpad/architecture_old_d128.py` (copy over `rwkv/architecture.py`,
eval, swap back), bf16 CUDA, `get_result`, by-user-mean LogLoss.
- **ahead: [PENDING — needs eval data for 5001–10000]**
- **imm:   [PENDING]**
- Goal: our 193k model trained on 1–5000 **matches or beats** these on the same 5001–10000 set.

## Our 5k runs (H=2/K=16, trained 1–5000, eval 5001–10000)
| run | HP (lr/warmup/wd/clip) | WS ep | decay ep | ahead | imm | vs baseline | notes |
|---|---|---|---|---|---|---|---|
| _(pending HP tune + data prep)_ | | 2 | 0.5 | | | | |

## HP tuning — DECIDED: on the FULL 5k dataset (Andrew 2026-06-30), deferred
**Decision:** tune HPs on the **full 5k** (train 1–5000, 2 WS + 0.5 decay), NOT the 1500-proxy. The
proxy probe showed 2 epochs OVERFIT 1500 users (baseline lost to the 1-epoch champion), and overfitting
scales with user count, so the proxy understates the 2-epoch budget at 5k → it is not a faithful
surrogate. The full-5k tune is slow (~10 h/trial, ~days/sweep) → **deferred until later** (needs the 5k
data prep first, and CPU is reserved for the sibling). `optimization/hp_tuner_5k.py` is reusable: it
keeps the 2 WS + 0.5 decay + H2K16 recipe; re-point its data paths to the 5k train_db + recompute
GROUPS_PER_EPOCH, and tune eval on a subset of 5001–10000 for speed. Levers = peak_lr, warmup,
weight_decay, clip; WS epochs FIXED at 2, decay FIXED at 0.5. Champion HP anchor: 1e-3 / 200 / 0.01 / 0.25.
The 1500-proxy run below was the (now-superseded) surrogate probe.
| trial | param | ahead | imm | obj | notes |
|---|---|---|---|---|---|
| hp5k_baseline | champion HPs | 0.318732 | 0.287316 | 0.606048 | 2 WS + 0.5 decay on 1500-proxy |

**★ FINDING (2026-06-30): 2 epochs on the 1500-proxy is WORSE than 1 epoch** (baseline 0.318732/0.287316
vs the 1-epoch champion 0.309723/0.276566 = +0.009 ahead / +0.011 imm). The same "variety beats
repetition" effect: revisiting 1500 users twice overfits to them, hurting 101-200 generalization. CAVEAT
for the proxy -- 2 epochs overfits MORE on 1500 users than on 5000, so the proxy understates the
2-epoch budget at true 5k scale. (Tuner STOPPED by Andrew after the baseline; resumable -- journal kept,
restart continues from trial 2 hp5k_peak_lr_0p0007.)

## Data prep (the long pole) — HARNESS READY + SMOKE-VALIDATED, DEFERRED (Andrew 2026-07-01)
**DECISION (Andrew 2026-07-01): fully DEFER the 5k data build until the sibling quantization research is
done; then run it with MORE threads (~4-6), NOT 1.** At 1 thread it's far too slow (see timing below).
Nothing launched. The build infra is written + smoke-tested and ready to fire with a thread bump.

**Scope chosen (Andrew): train + eval, BOTH halves.** Needed DBs (eval DBs currently cover only ~users
1-200, so the 5k ranges must be built):
- `train_db(1-5000)` sc8k -> **C:** (`train_db_5k_h1`, fast M.2, primary run reads every step)
- `train_db(5001-10000)` sc8k -> **F:** (`F:/rwkv_lmdb/train_db_5k_h2`, 4 TB USB; C: can't hold both)
- `test_db` (whole-user eval) both halves -> **F:** (`F:/rwkv_lmdb/test_db_5k`, users 1-10000)
- `label_filter` both halves -> extends the canonical **C:** `label_filter_db` (FSRS-6 --short --secs)

**Disk is NOT the constraint** (was the old worry): C: ~455 GB free, F: ~1237 GB free; lmdb `map_size` is
a SPARSE file on Windows (500 GB map -> 0 GB actual until written), so generous map_size is safe -- monitor
FREE space, not the logical file size. train_db ~51 MB/user (from `train_db_sc8k_1500`) -> ~255 GB/half.

**TIME is the constraint (why 1 thread was rejected).** Smoke rates (`scratchpad/run_build_5k.cmd` steps on
2-3 users): find_equalize ~0.42 ms/review, test_db ~0.32, train_db ~0.6-0.8; dataset = ~745M reviews. So at
**1 thread: full both-halves ~13 days; primary-only ~6 days.** At ~4-6 threads (like the old PROCESSES=7
builds): **~2-4 days.** => run threaded when the CPU frees up.

**READY-TO-RUN INFRA (all written + validated this session; just bump threads then launch):**
- 6 configs in `rwkv/`: `find_equalize_5k_{h2,h1}.toml`, `data_processing_test_5k_{h2,h1}.toml`,
  `data_processing_train_5k_{h1,h2}.toml`. **All have `PROCESSES = 1` -> change to 4-6 before launching.**
- Driver `scratchpad/run_build_5k.cmd`: runs the 6 builds sequentially, RESUMABLE (skips done users),
  continue-on-error, logs to `scratchpad/build_5k.log`. Order front-loads the 5001-10000 eval data (steps
  1-2) so the d=128 baseline eval can run while `train_db(1-5000)` builds.
- Launch detached (survives Esc): `powershell -NoProfile -File scratchpad/detach.ps1 -Script
  C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\run_build_5k.cmd`; monitor via OS truth (tail the log +
  FREE space on C:/F: + python PID). Smoke confirmed: configs parse, find_equalize runs, F: + C: writes work.
- [ ] (DEFERRED) run the harness with 4-6 threads once quant research frees the CPU

## Decisions / running notes
- 2026-06-30: phase opened. Speedups banked first: **Tier 1** (cudaMalloc→caching allocator, bit-exact,
  ~1.3–1.44× WKV microbench) + dead-array cleanup — pending in-place deploy once the export frees the
  CUDA `.pyd`. Tier 2 (occupancy) and Tier 3 (tensor cores) explored and dropped (kernel is
  latency-bound; tensor cores hit a wall at K=16). See `CLAUDE.md` / tasks for the full reasoning.
- 2026-07-01: **Tier 1 DEPLOYED in-place** — production `rwkv/model/RWKV_CUDA.cp312-win_amd64.pyd` is now
  byte-identical (SHA256 match) to the bit-exact-validated build. (Real-world WS steps/s A/B deferred to
  the next training run; correctness already established via the isolated-build golden.)
- 2026-07-01: **TENSOR CORES — profiled + KILLED (hard numbers).** `scratchpad/prof_wkv.py` (torch.profiler,
  per-kernel CUDA time, champion regime H=2/K=16) shows the ONLY matmuls (scan: `rwkv7_scan_kernel` +
  `rwkv7_add_kernel`) are **<=1.1% of WKV GPU time, dropping to 0.74% at B16xT30000** (the realistic 5k
  shape) -- the rest (95.9-97.5%) is the per-timestep matrix-VECTOR warp-shuffle recurrence, which tensor
  cores can't touch. Amdahl ceiling <1% => the low-risk "tensor-core the scan" win is DEAD, confirmed
  empirically (not just analytically). Time sinks: backward `final` ~61%, forward `final`/`base` ~12/11%,
  backward `base` ~11%. The ONLY path to TCs is a from-scratch **chunked-matmul (fla-style delta-rule)
  rewrite** of that 96% -- real but multi-day + parity-risky (+-0.0005 gate; K=16 underfills TC tiles).
  DEFERRED: revisit only if 5k runs prove painfully slow.
