@echo off
REM ===========================================================================================
REM PHASE 1, step 2 (v2): the CHAMPION MAX sweep, in REVIEWS per second.
REM
REM WHY. The champion bench measured peak_reserved_gb=13.069 on a 12.28 GB card, so every timing
REM this project owns at MAX=65536 was taken while the driver was paging. If a smaller MAX fits
REM under the ceiling it can be FASTER in reviews/s despite fewer reviews per step.
REM
REM ---- TWO THINGS v1 GOT WRONG, both fixed here, both worth stating ----
REM
REM 1. KD MUST BE OFF, AND NOT AS A PREFERENCE. The KD dump stores teacher logits keyed to the
REM    BATCH STREAM, and MAX *is* the batch stream. Every non-baseline arm of v1 aborted with
REM    exit 43 and "labels_sum 40756926045.9 vs dumped 124347298553.5, batch stream diverged".
REM    That is the alignment guard working exactly as designed. So a KD run can only ever be
REM    benched at its own MAX, and a sweep is necessarily a PLAIN-recipe measurement.
REM    Consequence for reading the result: KD costs a roughly per-REVIEW amount, so it dilutes
REM    the spread slightly but does not reorder it. Compare arms to each other, NEVER to the
REM    19528.6 rev/s KD number on record.
REM    The guard below refuses to start if KD is somehow still set -- a silent KD-on arm would
REM    abort at step 1 and look like a crash rather than a config error.
REM
REM 2. A DISPATCH-BOUND BENCHMARK NEEDS A QUIET CPU. v1's own 65536 arm read 16191 rev/s
REM    against 19529 on record at the same MAX -- a 17% deficit -- because the 6-thread e2s
REM    rebuild was running. The step is 85% CPU-bound, so a CPU co-tenant hits it directly.
REM    This runner therefore WAITS for the rebuild to finish before it times anything. The
REM    existing rule says no co-tenant GPU work during gate-critical runs; a dispatch-bound
REM    benchmark extends it to CPU.
REM
REM MEASUREMENT RULES, previously paid for: compare on reviews_per_sec whenever MAX differs;
REM 65536 runs first AND last as the drift control; each arm settles before it is timed.
REM
REM MAX is not a pure speed lever -- it sets the group count and so the optimizer steps per
REM epoch, which is why iter 34's move to 65536 cost 0.0003 at the old LR. This produces a SPEED
REM number. Adopting a MAX needs the LR retuned with it (phase 5) and re-bases the champion.
REM
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal enabledelayedexpansion
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\sweep_max2.log
set STAMP=%RANDOM%%RANDOM%
set SRC=scratchpad\iter53_muonlora\i53_ws.toml
set REBUILDLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\e2s_rebuild.log
set V1LOG=%DIR%\sweep_max.log

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
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
REM ---- THE FORCED CHANGE. Empty, not merely different. See note 1 in the header. ----
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120

echo ===== MAX SWEEP v2 START %DATE% %TIME% ===== >> "%LOG%"

REM ---- guard: KD really is off in the environment this runner will hand to python ----
if not "%RWKV_KD_MIX%"=="" (
  echo KD_STILL_SET -- refusing to sweep >> "%LOG%"
  echo DONE_EXIT_43 >> "%LOG%"
  exit /b 43
)

REM ---- phase 0: generate every toml up front ----
for %%M in (65536 49152 40960 32768 24576) do (
  .venv\Scripts\python.exe scratchpad/dispatch/write_max_toml.py %SRC% %%M "%DIR%\ws_max%%M.toml" >> "%LOG%" 2>&1
  if not !ERRORLEVEL!==0 (
    echo TOMLFAIL %%M >> "%LOG%"
    echo DONE_EXIT_31 >> "%LOG%"
    exit /b 31
  )
)

REM ---- phase 1: WAIT FOR A QUIET MACHINE. Anchored grep -- an unanchored one matches prose. ----
echo --- waiting for the e2s rebuild to finish %TIME% >> "%LOG%"
:waitrebuild
findstr /B /C:"DONE_EXIT_" "%REBUILDLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 61 127.0.0.1 >nul
  goto waitrebuild
)
echo --- rebuild finished %TIME% >> "%LOG%"

echo --- waiting for sweep v1 to finish %TIME% >> "%LOG%"
:waitv1
findstr /B /C:"DONE_EXIT_" "%V1LOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto waitv1
)
echo --- v1 finished, machine should be quiet %TIME% >> "%LOG%"

REM ---- the arms. 65536 first and last as the drift control. ----
for %%M in (65536 49152 40960 32768 24576 65536) do (
  ping -n 61 127.0.0.1 >nul
  echo --- bench MAX=%%M !TIME! >> "%LOG%"
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%DIR%\ws_max%%M.toml" > "%DIR%\v2_%%M_%STAMP%.log" 2>&1
  set RC=!ERRORLEVEL!
  findstr /C:"BENCH_RESULT" "%DIR%\v2_%%M_%STAMP%.log" >> "%LOG%"
  if not !RC!==0 echo   ARM_FAILED max=%%M exit=!RC! >> "%LOG%"
)

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
