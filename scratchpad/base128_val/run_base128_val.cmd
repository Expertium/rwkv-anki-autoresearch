@echo off
REM ============================================================================
REM OLD d=128 (2.76M) MODEL on the VAL half 5001-7500, AS INTENDED TO BE USED
REM (Andrew 2026-08-02): no rectifier, piecewise-linear ahead correction ON.
REM
REM That is the 2026-07-03 baseline's exact configuration, so this re-derives
REM ahead 0.294612 / imm 0.263561 under CURRENT code. Writing to NEW tags so the
REM historical jsonls are not clobbered and the two can be diffed -- if they
REM disagree, the phase's "0.0037 behind upstream" anchor has drifted and every
REM comparison built on it needs revisiting.
REM
REM ⚠ ARCH: uses RWKV_ARCH_MODULE, NOT the old file-copy swap. run_base5k_eval.cmd
REM copied architecture_old_d128.py over rwkv/architecture.py and copied it back --
REM which leaves the repo broken if it dies between the two. The env hook does the
REM same job with no mutable state.
REM
REM ⚠ EVERY A18 TRUNK FLAG MUST STAY UNSET. The d=128 model has no GRU head, no
REM state clamp, no cmix stripping, no feature mask, no Muon, and was trained WITH
REM feature 22 -- setting any of them would score a different model.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\base128_val
set LOG=%DIR%\base128_val.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py

REM --- explicitly cleared: everything that would make this a different model ---
set RWKV_EVAL_PAVA=
set RWKV_NO_AHEAD_RESIDUAL=
set RWKV_PAVA_LAMBDA=
set RWKV_PROBE_DENSITY=
set RWKV_PROBE_DUR=
set RWKV_GRU_HEAD=
set RWKV_MUON=
set RWKV_MUON_BATCHED=
set RWKV_STRIP_L0_VLORA=
set RWKV_ZERO_FEATURES=
set RWKV_STATE_CLAMP_TAU=
set RWKV_STATE_CLAMP_WINDOW=
set RWKV_STRIP_CMIX=
set RWKV_N_HEADS=
set RWKV_HEAD_DIM=
set RWKV_QAT_LOWRANK_SCOPE=
set RWKV_QAT_PQ=
set RWKV_QAT_SHIFT_PQ=
set RWKV_QAT_FUSED=
set RWKV_QAT_SHIFT_SCOPE=
set RWKV_QAT_COMPILE=
set RWKV_NO_JIT=

echo ===== BASE128 VAL EVAL (5001-7500, no rectifier, residual ON) START %DATE% %TIME% ===== > "%LOG%"

REM No del: eval_sharded skips users already banked, so a retry only re-risks the remainder.
echo === EVAL attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/base128_val/base128_val.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/base128_val/base128_val.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === COMPARE vs the 2026-07-03 run and vs the current champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/base128_val/compare_base128.py >> "%LOG%" 2>&1
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
