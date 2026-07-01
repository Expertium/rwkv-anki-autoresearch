# RWKV 5k phase — working notes (supplement to `research_5k.md`)

`research_5k.md` is the human-facing front (one results table only). All detail, reasoning, and
running notes live here. Pre-5k history: `research_log.md` / `log.md` / `HISTORY.md`.

**Front-table conventions:** LogLoss to 4 decimal places; parameter counts exact (e.g. 2,762,884 /
193,724, from `optimization/model_stats.py`). Working precision here in the notes may be higher.

## Setup
- **Train** users 1–5000; **eval** users 5001–10000 (disjoint held-out half).
- **Compute budget:** 2 WS epochs + 0.5 decay epochs (cosine).
- **Model:** H=2/K=16 champion (d=32, 2 heads × K=16, layers [1,4,3,3,3], 193,724 params, per-card
  WKV state = two 16×16 per-head matrices). Env: `RWKV_N_HEADS=2 RWKV_HEAD_DIM=16`,
  `RWKV_EMPTY_CACHE_EVERY=0`, `RWKV_DETERMINISTIC=1`, `RWKV_AUGMENT_SEED=1234`, HP from the tuner.

## Baseline to beat
The original leaderboard d=128 model `pretrain/RWKV_trained_on_101_4999.pth` (2.76M params, 4 heads ×
K=32), eval on 5001–10000 (genuine held-out — it trained on 101–4999). Eval via arch-swap
`scratchpad/architecture_old_d128.py` (copy over `rwkv/architecture.py`, eval, swap back), bf16 CUDA,
`get_result`, by-user-mean LogLoss. PENDING — needs the 5001–10000 eval data. Goal: our 194k model
trained on 1–5000 matches/beats it on the same set.

## HP tuning — tune on the FULL 5k, deferred (Andrew 2026-06-30)
Tune HPs on the full 5k (train 1–5000, 2 WS + 0.5 decay), NOT the 1500-proxy. Levers: peak_lr, warmup,
weight_decay, clip; WS epochs fixed at 2, decay fixed at 0.5. Champion HP anchor: 1e-3 / 200 / 0.01 / 0.25.
`optimization/hp_tuner_5k.py` is reusable — re-point its data paths to the 5k train_db, recompute
GROUPS_PER_EPOCH, tune-eval on a subset of 5001–10000.
- **FINDING (2026-06-30): 2 epochs on the 1500-proxy is WORSE than 1 epoch** — proxy baseline (champion
  HPs, 2 WS + 0.5 decay on 1500 users) = 0.318732 / 0.287316 vs the 1-epoch champion 0.309723 / 0.276566
  (+0.009 ahead / +0.011 imm). "Variety beats repetition": revisiting 1500 users twice overfits. And the
  proxy overfits MORE on 1500 than on 5000, so it understates the 2-epoch budget at true 5k scale → the
  proxy is not a faithful surrogate. (Tuner stopped after the baseline; resumable from trial 2.)

## Data prep — HARNESS READY + SMOKE-VALIDATED, DEFERRED (Andrew 2026-07-01)
Fully defer the 5k data build until the sibling quantization research frees the CPU, then run it with
**more threads (~4–6), NOT 1**. Nothing launched. Scope (Andrew): train + eval, BOTH halves.

DBs to build (eval DBs currently cover only ~users 1–200):
- `train_db(1-5000)` sc8k → **C:** (`train_db_5k_h1`, fast M.2, primary run reads every step)
- `train_db(5001-10000)` sc8k → **F:** (`F:/rwkv_lmdb/train_db_5k_h2`, 4 TB USB; C: can't hold both)
- `test_db` (whole-user) both halves → **F:** (`F:/rwkv_lmdb/test_db_5k`, users 1–10000)
- `label_filter` both halves → extends the canonical **C:** `label_filter_db` (FSRS-6 --short --secs)

Disk is NOT the constraint: C: ~455 GB free, F: ~1237 GB free; lmdb `map_size` is a SPARSE file on
Windows (500 GB map → 0 GB actual until written) — monitor FREE space, not logical file size. train_db
~51 MB/user → ~255 GB/half.

TIME is the constraint (why 1 thread was rejected). Smoke rates: find_equalize ~0.42 ms/review, test_db
~0.32, train_db ~0.6–0.8; dataset ~745M reviews → at 1 thread full both-halves ~13 days / primary-only
~6 days; at 4–6 threads ~2–4 days.

Ready-to-run infra (just bump threads then launch):
- 6 configs in `rwkv/`: `find_equalize_5k_{h2,h1}.toml`, `data_processing_test_5k_{h2,h1}.toml`,
  `data_processing_train_5k_{h1,h2}.toml`. All have `PROCESSES = 1` → change to 4–6 before launching.
- Driver `scratchpad/run_build_5k.cmd`: 6 builds sequentially, RESUMABLE (skips done users),
  continue-on-error, logs to `scratchpad/build_5k.log`. Order front-loads the 5001–10000 eval data
  (steps 1–2) so the d=128 baseline eval can run while `train_db(1-5000)` builds.
- Launch detached (survives Esc): `powershell -NoProfile -File scratchpad/detach.ps1 -Script
  C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\run_build_5k.cmd`; monitor via OS truth (tail the
  log + FREE space on C:/F: + python PID). Smoke confirmed: configs parse, find_equalize runs, F:+C: writes work.

## Speedups banked (detail also in CLAUDE.md)
- 2026-07-01 **Tier 1 DEPLOYED in-place** — production `rwkv/model/RWKV_CUDA.cp312-win_amd64.pyd` is
  byte-identical (SHA256) to the bit-exact-validated build (cudaMalloc/cudaFree → caching-allocator
  scan scratch; ~1.3–1.44× WKV microbench). Real-world WS steps/s A/B deferred to the next training run.
- 2026-07-01 **Tensor cores profiled + KILLED** (`scratchpad/prof_wkv.py`). Only matmuls (scan) are
  ≤1.1% of WKV GPU time, 0.74% at B16×T30000; the other ~96% is per-timestep matrix-VECTOR warp-shuffle
  recurrence (backward `final` ~61%, fwd `final`/`base` ~12/11%, bwd `base` ~11%). Amdahl ceiling <1% →
  cheap tensor-core win DEAD. Only path to TCs = from-scratch chunked-matmul (fla delta-rule) rewrite of
  the recurrence — multi-day + parity-risky (±0.0005 gate; K=16 underfills TC tiles). Revisit only if 5k
  proves painfully slow.
