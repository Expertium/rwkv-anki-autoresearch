@echo off
REM ===========================================================================================
REM PHASE 1, step 2: the CHAMPION MAX sweep, measured in REVIEWS per second.
REM
REM WHY. The champion bench measured peak_reserved_gb=13.069 on a 12.28 GB card. Every timing
REM this project has recorded at MAX=65536 was therefore taken while the driver was paging, so
REM the recorded 0.907 steps/s is a paging number, not a compute number. If a smaller MAX fits
REM under the ceiling it can be FASTER in reviews/s despite processing fewer reviews per step.
REM That is the whole hypothesis, and it is worth measuring before any fusion work, because it
REM would change what the fusion work is measured against.
REM
REM METHOD, all of it previously paid for:
REM   - compare on reviews_per_sec, NEVER steps_per_sec, because MAX differs between arms.
REM   - 65536 runs TWICE, first and last, as an in-sweep control. The 19528.6 on record was
REM     measured in another session; without a repeat, drift is indistinguishable from effect.
REM   - each arm settles before it is timed. A hot card reads as a slow arm.
REM   - 120 warmup steps then 100 measured, the same window the champion bench used.
REM
REM WHAT THIS IS NOT. MAX is not a pure speed lever: it sets the group count and therefore the
REM number of optimizer steps per epoch, which is why iter 34's move to 65536 cost 0.0003 at the
REM old LR. A sweep result is a SPEED measurement. Adopting a different MAX needs the LR retuned
REM with it, which is phase 5, and it re-bases the champion.
REM
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal enabledelayedexpansion
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\sweep_max.log
set STAMP=%RANDOM%%RANDOM%
set SRC=scratchpad\iter53_muonlora\i53_ws.toml

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
set RWKV_KD_MIX=C:\rwkv_kd_dump\t128_seedpair_65k:10935
set RWKV_KD_ALPHA=0.9
set RWKV_FSRS_CARD=

set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120

echo ===== MAX SWEEP START %DATE% %TIME% ===== >> "%LOG%"

REM ---- phase 0: generate every toml up front, so a generator failure costs seconds not an hour
for %%M in (65536 49152 40960 32768 24576) do (
  .venv\Scripts\python.exe scratchpad/dispatch/write_max_toml.py %SRC% %%M "%DIR%\ws_max%%M.toml" >> "%LOG%" 2>&1
  if not !ERRORLEVEL!==0 (
    echo TOMLFAIL %%M >> "%LOG%"
    echo DONE_EXIT_31 >> "%LOG%"
    exit /b 31
  )
)

REM ---- the arms. 65536 first and last as the drift control. ----
for %%M in (65536 49152 40960 32768 24576 65536) do (
  ping -n 31 127.0.0.1 >nul
  echo --- bench MAX=%%M %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%DIR%\ws_max%%M.toml" > "%DIR%\sw_%%M_%STAMP%.log" 2>&1
  set RC=!ERRORLEVEL!
  findstr /C:"BENCH_RESULT" "%DIR%\sw_%%M_%STAMP%.log" >> "%LOG%"
  if not !RC!==0 echo   ARM_FAILED max=%%M exit=!RC! >> "%LOG%"
)

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
