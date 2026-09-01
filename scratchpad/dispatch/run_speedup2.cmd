@echo off
REM ===========================================================================================
REM PHASE 1 SPEEDUP, ROUND 2: size the DETERMINISM TAX on the indexing kernels.
REM
REM Round 1 (quiet machine) found the step is GPU-BOUND, not dispatch-bound: 1,368.84 ms/step
REM of GPU kernel time against DISPATCH_PLAN's recorded 237. And the top two kernels are
REM aten::_index_put_impl_ (18.95%) plus indexing_backward_kernel (18.67%) = 37.6% of GPU
REM time, about 515 ms/step -- more than the WKV recurrence and every matmul combined.
REM
REM RWKV_DETERMINISTIC=1 forces SORT-BASED scatter for accumulating index_put instead of
REM atomics, which is precisely that kernel pair. A 1.5x was banked attacking this once
REM (PermGather, 2026-07-03) leaving a ~57 ms tax, but the trunk has since gained the
REM interleaved schedule, which scatters and gathers per stream per round across 13
REM layer-steps. So the tax is re-measured rather than quoted.
REM
REM ⚠ A MEASUREMENT, NOT A PROPOSED ADOPTION. det=0 uses atomics and is NOT bit-reproducible;
REM every speedup banked so far has been bit-exact. This sizes what is RECOVERABLE, and so
REM whether a deterministic-but-fast path is worth hunting. A small tax would instead mean the
REM indexing cost is intrinsic and the interleave scatter itself is the next lever.
REM
REM empty_cache is PINNED at 1 in every arm: round 1 measured every=0 at 26% SLOWER with GPU
REM kernel time 1,369 -> 4,321 ms/step, confirming the allocator-creep rule.
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal enabledelayedexpansion
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\speedup2.log
set STAMP=%RANDOM%%RANDOM%
set CFG=%DIR%\swup_65536.toml

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_MUON_INCLUDE_LORA=1
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_FSRS_CARD=
set RWKV_ID_FEATURES=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

echo ===== SPEEDUP ROUND 2 START %DATE% %TIME% ===== >> "%LOG%"

REM ---- PHASE A: the clean re-profile, current config (every=1) ----
set RWKV_DETERMINISTIC=1
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_MAX_STEPS=
set RWKV_BENCH_WARMUP=
set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=5
echo --- PHASE A: baseline profile, deterministic=1 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\prof_det1_%STAMP%.log" 2>&1
findstr /C:"total GPU kernel time" "%DIR%\prof_det1_%STAMP%.log" >> "%LOG%"
echo   profile exit=!ERRORLEVEL! %TIME% >> "%LOG%"

REM ---- PHASE B: the A/B, alternating so drift is visible ----
set RWKV_PROFILE_STEP=
set RWKV_PROFILE_COUNT=
set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120
for %%E in (1 0 1 0) do (
  ping -n 31 127.0.0.1 >nul
  set RWKV_DETERMINISTIC=%%E
  set RWKV_EMPTY_CACHE_EVERY=1
  echo --- bench deterministic=%%E !TIME! >> "%LOG%"
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\det%%E_%STAMP%.log" 2>&1
  set RC=!ERRORLEVEL!
  findstr /C:"BENCH_RESULT" "%DIR%\det%%E_%STAMP%.log" >> "%LOG%"
  if not !RC!==0 (
    echo   ARM_FAILED deterministic=%%E exit=!RC! >> "%LOG%"
    findstr /C:"out of memory" "%DIR%\det%%E_%STAMP%.log" >nul 2>&1
    if not errorlevel 1 echo   ...CUDA OOM -- the creep concern is REAL and the flag stays >> "%LOG%"
  )
)

REM ---- PHASE C: profile again with every=0, to see where the time moved ----
set RWKV_DETERMINISTIC=0
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_MAX_STEPS=
set RWKV_BENCH_WARMUP=
set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=5
echo --- PHASE C: profile, deterministic=0 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\prof_det0_%STAMP%.log" 2>&1
findstr /C:"total GPU kernel time" "%DIR%\prof_det0_%STAMP%.log" >> "%LOG%"
echo   profile exit=!ERRORLEVEL! %TIME% >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
