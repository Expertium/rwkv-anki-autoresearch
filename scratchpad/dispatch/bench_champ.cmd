@echo off
REM ===========================================================================================
REM PHASE 1, step 1: re-measure the CHAMPION trunk before optimizing anything.
REM
REM Two questions, two arms, on the iter-53 recipe exactly (no arm promoted, so d=80 is still
REM the trunk the speedup must target).
REM
REM   ARM bench   -- steps/s, reviews/s and PEAK RESERVED VRAM at the production MAX=65536.
REM                  V1's timing measured peak_reserved_gb=12.583 on a 12 GB card. If the
REM                  CHAMPION is also near the ceiling then WDDM paging is depressing every run
REM                  we have ever timed, which would be a bigger lever than any fusion work.
REM                  CLAUDE.md already records the QAT config at 12.807 GB and flags exactly this.
REM   ARM profile -- the dispatch breakdown, to confirm the 85%-CPU-bound split still holds at
REM                  this arch and to re-derive the Amdahl ceiling. TRAINING_SPEED.md's numbers
REM                  predate iter 53.
REM
REM METHOD RULES, all previously paid for:
REM   - compare arms on reviews_per_sec, never steps_per_sec, whenever MAX differs.
REM   - do not benchmark immediately after a long GPU job; the card must settle.
REM   - a measurement taken above about 11 GB is suspect, which is the point of arm 1.
REM
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\bench_champ.log
set STAMP=%RANDOM%%RANDOM%

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

echo ===== CHAMPION BENCH START %DATE% %TIME% ===== >> "%LOG%"

REM let the card settle after the V1 sweep before timing anything
ping -n 91 127.0.0.1 >nul

REM ---- ARM 1: throughput + peak VRAM at the production MAX ----
set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120
echo --- bench MAX=65536 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter53_muonlora/i53_ws.toml > "%DIR%\bench_%STAMP%.log" 2>&1
findstr /C:"BENCH_RESULT" "%DIR%\bench_%STAMP%.log" >> "%LOG%"
if not %ERRORLEVEL%==0 (
  echo BENCH_FAILED - see bench_%STAMP%.log >> "%LOG%"
  echo DONE_EXIT_21 >> "%LOG%"
  exit /b 21
)

REM ---- ARM 2: the dispatch breakdown ----
set RWKV_MAX_STEPS=
set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=5
echo --- profile at step 150 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter53_muonlora/i53_ws.toml > "%DIR%\profile_%STAMP%.log" 2>&1
echo   profile exit=%ERRORLEVEL% %TIME% >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
