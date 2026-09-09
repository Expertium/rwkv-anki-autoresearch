@echo off
REM ENDGAME ARM 1, PHASE C ONLY -- the rectified VAL-half eval.
REM
REM WHY THIS EXISTS. run_w10plain.cmd ran its 6.2 h decay to completion (w10p_d_21870.pth, 23:37)
REM and then killed itself at 23:38 with DONE_EXIT_28 = DECAY_SHORT, because its artifact guard
REM tested "%DIR%\w10p_d_21870.pth" -- i.e. scratchpad\w10plain -- while a DECAY-ONLY run writes
REM its checkpoints into the SOURCE run's folder, scratchpad\ws10. Nothing was lost; the decay is
REM banked.
REM
REM !! THE HONEST PART: I found this bug class earlier today and fixed only HALF of it. The
REM 2026-09-09 patch taught write_eval_toml.py to fall back to the run's own decay.toml, which
REM covers PHASE C. The identical wrong-directory test in the PHASE B artifact guard was left
REM alone. A fix that repairs one consumer of a wrong path and not the other is not a fix.
REM Proven before this file was armed: the phase-C toml step now resolves
REM   [ckpt-fallback] none in scratchpad/w10plain; decay.toml points at scratchpad/ws10, found 12
REM   wrote ...  to  scratchpad/ws10/w10p_d_21870.pth (step 21870)
REM
REM Env is copied VERBATIM from run_w10plain.cmd. Fresh log: w10plain.log already carries a
REM terminal marker, and a downstream waiter must never poll a log that does.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10plain
set SRC=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ws10
set LOG=%DIR%\w10plain_eval.log
set STAMP=%RANDOM%%RANDOM%
set TAG=w10plain
set STEPS=21870

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_DECAY_VALIDATE_EVERY=2000
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
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
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1

set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
set RWKV_ZERO_FEATURES=
set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_id5
set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_id5
set RWKV_LABEL_FILTER_DB=F:/rwkv_lmdb/label_filter_db_id_e2s
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

echo ===== w10plain EVAL START %DATE% %TIME% ===== > "%LOG%"

REM ---- PHASE 0: the decay artifact, tested WHERE A DECAY-ONLY RUN ACTUALLY WRITES IT ----
if not exist "%SRC%\w10p_d_%STEPS%.pth" (
  echo %TAG% DECAY_CKPT_MISSING %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_28 %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
REM ---- PHASE 0b: KD must have been ABSENT from the decay that produced it ----
findstr /C:"[kd-mix] KD ON" "%DIR%\decay_587824236.log" >nul
if %ERRORLEVEL%==0 (
  echo %TAG% KD_LEAKED_IN %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_43 %DATE% %TIME% >> "%LOG%"
  exit /b 43
)
echo %TAG% DECAY_OK %TIME% >> "%LOG%"

REM ---- PHASE C: rectified VAL-half eval, users 5001-7500 ----
set RWKV_EVAL_PAVA=1
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/w10plain w10p_d %DIR%\eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% ETOML_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_24 %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
findstr /C:"F:/rwkv_lmdb/test_db_5k_id5" "%DIR%\eval.toml" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_DB_MISMATCH %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_34 %DATE% %TIME% >> "%LOG%"
  exit /b 34
)
REM And it must score the FINAL decay checkpoint, not an intermediate one. The fallback picks
REM the highest step it finds, so this is the assertion that the pick was right.
findstr /C:"w10p_d_%STEPS%.pth" "%DIR%\eval.toml" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_WRONG_CKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_35 %DATE% %TIME% >> "%LOG%"
  exit /b 35
)
if exist "result\RWKV-%TAG%.jsonl" del /q "result\RWKV-%TAG%.jsonl"
if exist "result\RWKV-P-%TAG%.jsonl" del /q "result\RWKV-P-%TAG%.jsonl"
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\eval.toml --shards 1 --solo-threshold 0 --exclude 6701 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_25 %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo %TAG% EVAL_OK %TIME% >> "%LOG%"

REM Terminal marker BEFORE endlocal.
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
