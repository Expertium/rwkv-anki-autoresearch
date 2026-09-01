@echo off
REM ===========================================================================================
REM PHASE 1 SPEEDUP, ROUND 1: a CLEAN profile, then the allocator A/B.
REM
REM Andrew 2026-08-31: "after re-basing look for speedups (both GPU-related and CPU
REM overhead-related)". The re-base is done, so this is the priority. He also noted that another
REM Claude was hammering the CPU when the 2026-08-30 profile was taken -- which matters, because
REM that profile is the evidence base:
REM
REM   * CPU self-times (cudaFree 581.7, cudaMalloc 338.1, cudaLaunchKernel 303.8 ms/step) are
REM     host-side waits and WOULD be inflated by a busy CPU.
REM   * total GPU kernel time (1,416.74 ms/step) is measured ON DEVICE and would NOT be.
REM
REM That second number is the one that contradicts DISPATCH_PLAN's headline of 237 ms/step, i.e.
REM the whole "the step is 85% dispatch-bound, CUDA graphs are the top candidate" thesis. So the
REM first job is to re-take the profile on a quiet machine and find out which figure is real.
REM
REM ---- THE LEAD BEING TESTED ----
REM `RWKV_EMPTY_CACHE_EVERY=1` calls torch.cuda.empty_cache() every step, which is why cudaFree
REM shows ~395 calls/step. It is set for d=80 runs because of "allocator creep -> WDDM paging ->
REM 4x slowdown". THAT PREMISE IS FALSIFIED: sampling real training gave a peak of 8,182 MiB on a
REM 12,282 MiB card, roughly 4 GB spare. The record already measured 1.12x from every=0 on the old
REM d=32 config with no OOM.
REM
REM peak_reserved_gb in each BENCH_RESULT is the safety readout: if every=0 lets the allocator
REM creep toward the card, it shows up there and the flag stays as it is.
REM
REM METHOD: arms alternate 1/0/1/0 so drift is visible and the comparison is paired twice.
REM Plain recipe (KD off) throughout -- allocator behaviour is recipe-independent, and it keeps
REM these numbers comparable to the MAX sweeps.
REM ⚠ The toml saves into scratchpad/dispatch/benchout, NOT into a run directory.
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal enabledelayedexpansion
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\speedup1.log
set STAMP=%RANDOM%%RANDOM%
set CFG=%DIR%\swup_65536.toml

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
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

echo ===== SPEEDUP ROUND 1 START %DATE% %TIME% ===== >> "%LOG%"

REM ---- PHASE A: the clean re-profile, current config (every=1) ----
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_MAX_STEPS=
set RWKV_BENCH_WARMUP=
set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=5
echo --- PHASE A: clean profile, empty_cache=1 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\prof_ec1_%STAMP%.log" 2>&1
findstr /C:"total GPU kernel time" "%DIR%\prof_ec1_%STAMP%.log" >> "%LOG%"
echo   profile exit=!ERRORLEVEL! %TIME% >> "%LOG%"

REM ---- PHASE B: the A/B, alternating so drift is visible ----
set RWKV_PROFILE_STEP=
set RWKV_PROFILE_COUNT=
set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120
for %%E in (1 0 1 0) do (
  ping -n 31 127.0.0.1 >nul
  set RWKV_EMPTY_CACHE_EVERY=%%E
  echo --- bench empty_cache=%%E !TIME! >> "%LOG%"
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\ec%%E_%STAMP%.log" 2>&1
  set RC=!ERRORLEVEL!
  findstr /C:"BENCH_RESULT" "%DIR%\ec%%E_%STAMP%.log" >> "%LOG%"
  if not !RC!==0 (
    echo   ARM_FAILED empty_cache=%%E exit=!RC! >> "%LOG%"
    findstr /C:"out of memory" "%DIR%\ec%%E_%STAMP%.log" >nul 2>&1
    if not errorlevel 1 echo   ...CUDA OOM -- the creep concern is REAL and the flag stays >> "%LOG%"
  )
)

REM ---- PHASE C: profile again with every=0, to see where the time moved ----
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_MAX_STEPS=
set RWKV_BENCH_WARMUP=
set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=5
echo --- PHASE C: profile, empty_cache=0 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\prof_ec0_%STAMP%.log" 2>&1
findstr /C:"total GPU kernel time" "%DIR%\prof_ec0_%STAMP%.log" >> "%LOG%"
echo   profile exit=!ERRORLEVEL! %TIME% >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
