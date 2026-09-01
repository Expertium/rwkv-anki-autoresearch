@echo off
REM ===========================================================================================
REM PHASE 1: the UPWARD MAX sweep. About 30 minutes.
REM
REM WHY UPWARD, WHEN THE DOWNWARD SWEEP JUST CLOSED. The whole phase opened on
REM peak_reserved_gb=13.069 against a 12.28 GB card, read as "we are paging". Sampling a REAL
REM training phase says otherwise: 171 nvidia-smi samples over 75 s peaked at 8,182 MiB, roughly
REM 4 GB of headroom. peak_reserved is torch.cuda.max_memory_reserved(), a high-water mark that
REM empty_cache does not reset, so it reported a transient forever; nvidia-smi reports what the
REM driver actually handed out, which is what governs paging.
REM
REM So MAX may have been left at 65536 for a constraint that never applied, and the downward
REM result (65536 beat every smaller value on reviews/s, VRAM flat within 1%) is exactly what a
REM non-binding memory limit looks like. Upward is the untested direction.
REM
REM AN OOM IS A RESULT, NOT A FAILURE. If 114688 does not fit, the arm logs ARM_FAILED and the
REM loop continues -- that IS the answer to "how much headroom is there".
REM
REM METHOD, all previously paid for: compare on reviews_per_sec, never steps_per_sec, because MAX
REM differs; 65536 runs FIRST and LAST as the drift control; each arm settles before timing; KD is
REM forced OFF because the dump is keyed to the batch stream and MAX *is* the batch stream, so a
REM KD run can only ever be benched at its own MAX.
REM
REM ⚠ The tomls save into scratchpad/dispatch/benchout with prefix swup<MAX>, NOT into the
REM re-base's directory. A bench cloned from ws.toml would otherwise overwrite e2sc_ws checkpoints.
REM
REM MAX is not a pure speed lever -- it sets the group count and so the optimizer steps per epoch
REM (iter 34's move to 65536 cost 0.0003 at the old LR). This produces a SPEED number only.
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal enabledelayedexpansion
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\sweep_up.log
set STAMP=%RANDOM%%RANDOM%
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\e2s_rebase\e2s_rebase.log

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
set RWKV_ID_FEATURES=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120

echo ===== UPWARD MAX SWEEP armed %DATE% %TIME% ===== >> "%LOG%"

REM ---- wait for the e2s re-base to release the GPU ----
:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo re-base finished, GPU free %DATE% %TIME% >> "%LOG%"

REM ---- guard: KD really is off in the environment handed to python ----
if defined RWKV_KD_MIX (
  echo KD_STILL_SET -- refusing to sweep >> "%LOG%"
  echo DONE_EXIT_43 >> "%LOG%"
  exit /b 43
)

REM ---- the arms. 65536 first and last as the drift control. ----
for %%M in (65536 81920 98304 114688 65536) do (
  ping -n 31 127.0.0.1 >nul
  echo --- bench MAX=%%M !TIME! >> "%LOG%"
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%DIR%\swup_%%M.toml" > "%DIR%\up_%%M_%STAMP%.log" 2>&1
  set RC=!ERRORLEVEL!
  findstr /C:"BENCH_RESULT" "%DIR%\up_%%M_%STAMP%.log" >> "%LOG%"
  if not !RC!==0 (
    echo   ARM_FAILED max=%%M exit=!RC! -- if this is an OOM, that IS the headroom answer >> "%LOG%"
    findstr /C:"out of memory" "%DIR%\up_%%M_%STAMP%.log" >nul 2>&1
    if not errorlevel 1 echo   ...confirmed CUDA out of memory at MAX=%%M >> "%LOG%"
  )
)

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
