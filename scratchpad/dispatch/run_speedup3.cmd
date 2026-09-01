@echo off
REM ===========================================================================================
REM PHASE 1 SPEEDUP, ROUND 3: WHAT DOES THE INTERLEAVED SCHEDULE COST IN GPU TIME?
REM
REM THE LEAD, and it explains a discrepancy rather than just chasing one. train_rwkv.py's own
REM profiler comment records 237 ms of GPU kernel time in a ~1450 ms step -- dated 2026-07-27.
REM Interleaving landed with iter 41 on 2026-08-11. So that figure was measured on a d=80 trunk
REM WITHOUT the interleaved schedule. It was not wrong; it is STALE. Rounds 1-2 measure
REM 1,206-1,369 ms/step on the current trunk, and the thing added in between is 13 layer-steps
REM of per-stream gather and scatter -- exactly the indexing that is now 37.6% of GPU time.
REM
REM So the hypothesis is: interleaving is responsible for most of the gap. This measures it.
REM
REM ⚠ NOT A PROPOSED REVERT. Interleaving is an ACCEPTED accuracy win (iter 41, +0.000216 to
REM +0.000611 in both modes) and the protocol explicitly leaves GPU training speed untimed. The
REM point is to PRICE it: if it costs ~5x the step time for that gain, the trade is worth knowing
REM and the optimisation target becomes making the schedule cheap, not removing it.
REM
REM Pinned: determinism=1 and empty_cache=1, both settled by rounds 1-2. Arms alternate 1/0/1/0.
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal enabledelayedexpansion
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\speedup3.log
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

echo ===== SPEEDUP ROUND 3 START %DATE% %TIME% ===== >> "%LOG%"

REM ---- PHASE A: the clean re-profile, current config (every=1) ----
set RWKV_DETERMINISTIC=1
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_MAX_STEPS=
set RWKV_BENCH_WARMUP=
set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=5
echo --- PHASE A: baseline profile, interleave=1 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\prof_ilv1_%STAMP%.log" 2>&1
findstr /C:"total GPU kernel time" "%DIR%\prof_ilv1_%STAMP%.log" >> "%LOG%"
echo   profile exit=!ERRORLEVEL! %TIME% >> "%LOG%"

REM ---- PHASE B: the A/B, alternating so drift is visible ----
set RWKV_PROFILE_STEP=
set RWKV_PROFILE_COUNT=
set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120
for %%E in (1 0 1 0) do (
  ping -n 31 127.0.0.1 >nul
  set RWKV_INTERLEAVE=%%E
  set RWKV_DETERMINISTIC=1
  set RWKV_EMPTY_CACHE_EVERY=1
  echo --- bench interleave=%%E !TIME! >> "%LOG%"
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\ilv%%E_%STAMP%.log" 2>&1
  set RC=!ERRORLEVEL!
  findstr /C:"BENCH_RESULT" "%DIR%\ilv%%E_%STAMP%.log" >> "%LOG%"
  if not !RC!==0 (
    echo   ARM_FAILED interleave=%%E exit=!RC! >> "%LOG%"
    findstr /C:"out of memory" "%DIR%\ilv%%E_%STAMP%.log" >nul 2>&1
    if not errorlevel 1 echo   ...CUDA OOM -- the creep concern is REAL and the flag stays >> "%LOG%"
  )
)

REM ---- PHASE C: profile again with every=0, to see where the time moved ----
set RWKV_INTERLEAVE=0
set RWKV_DETERMINISTIC=1
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_MAX_STEPS=
set RWKV_BENCH_WARMUP=
set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=5
echo --- PHASE C: profile, interleave=0 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\prof_ilv0_%STAMP%.log" 2>&1
findstr /C:"total GPU kernel time" "%DIR%\prof_ilv0_%STAMP%.log" >> "%LOG%"
echo   profile exit=!ERRORLEVEL! %TIME% >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
