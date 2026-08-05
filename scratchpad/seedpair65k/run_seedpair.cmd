@echo off
REM ===========================================================================================
REM SEED PAIR (Andrew 2026-07-30: "let's do iter 31 and iter 32 both with the same seed after
REM HP tune") -- run 2026-08-05 on the ITER-34 TUNED recipe, both arms at RWKV_AUGMENT_SEED=4321:
REM   DUMP : d=128 teacher walks the seed-4321 MAX=65536 stream -> C:\rwkv_kd_dump\t128_seedpair_65k
REM          (NEW dir on purpose -- the old t128_iter32 is the only artifact reproducing iter 32
REM          as accepted, and this file rmdir's %DUMP%; reusing the name would destroy it)
REM   ARM A: sp4321_nokd = tuned recipe, no KD   (the iter-31-lineage arm)
REM   ARM B: sp4321_kd   = tuned recipe + KD 0.5 (the iter-32-lineage arm)
REM Each arm: WS 10935 -> decay ratio 1.0 -> RECTIFIED eval on the FULL VAL half 5001-7500.
REM The quantity in doubt is (B - A) WITHIN seed 4321: does KD still win at a second seed, on
REM the tuned recipe? The pair also re-checks iter 34's own margin (its warmup/dropout adoptions
REM were inside noise): arm A vs the seed-1234 champion is the cross-seed spread of the recipe.
REM ⚠ KD applies to WS ONLY (iter 32 cleared it before decay; same here).
REM ⚠ NO del of result jsonls anywhere in this file -- tags are fresh (verified before writing),
REM   and eval retries resume from banked users (the giant-user OOM rule).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\seedpair65k
set LOG=%DIR%\seedpair.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set SMOKE=C:\rwkv_kd_dump\t128_seedpair_65k_smoke
set WSSTEPS=10935

echo ===== SEED PAIR START %DATE% %TIME% ===== > "%LOG%"

REM ------------------------------------------------------- TEACHER DUMP (d=128, forward only)
setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
REM The teacher is the ORIGINAL model: its own arch, none of the trunk flags. Cleared
REM explicitly (this .cmd may run in a shell where they linger).
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py
set RWKV_GRU_HEAD=
set RWKV_PAVA_LAMBDA=
set RWKV_MUON=
set RWKV_MUON_LR=
set RWKV_MUON_MOMENTUM=
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_DROPOUT_SCALE=
set RWKV_NO_AHEAD_RESIDUAL=
set RWKV_STRIP_L0_VLORA=
set RWKV_ZERO_FEATURES=
set RWKV_STATE_CLAMP_TAU=
set RWKV_STATE_CLAMP_WINDOW=
set RWKV_STRIP_CMIX=
set RWKV_VPRUNE_REF=
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=
REM Probe vars are DATA-side: teacher and student must agree on row layout.
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_KD_TEACHER=pretrain/RWKV_trained_on_101_4999.pth

echo === SMOKE DUMP 5 steps %TIME% === >> "%LOG%"
if exist "%SMOKE%" rmdir /s /q "%SMOKE%"
mkdir "%SMOKE%" 2>nul
set RWKV_KD_DUMP_OUT=%SMOKE%
set RWKV_KD_STEPS=5
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/seedpair65k/spdump.toml > "%DIR%\smoke_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SMOKEFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)
.venv\Scripts\python.exe scratchpad/iter32_kd/check_dump.py "%SMOKE%" --expect-steps 5 --planned-steps %WSSTEPS% --max-gb 60 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SMOKECHECK_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 10
)
rmdir /s /q "%SMOKE%"
echo SMOKE OK %TIME% >> "%LOG%"

echo === FULL DUMP %WSSTEPS% steps %TIME% === >> "%LOG%"
if exist "%DUMP%" rmdir /s /q "%DUMP%"
mkdir "%DUMP%" 2>nul
set RWKV_KD_DUMP_OUT=%DUMP%
set RWKV_KD_STEPS=%WSSTEPS%
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/seedpair65k/spdump.toml > "%DIR%\dump_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DUMPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 11
)
if not exist "%DUMP%\step_%WSSTEPS%.pt" (
  echo DONE_EXIT_DUMPSHORT %DATE% %TIME% >> "%LOG%"
  exit /b 12
)
echo DUMP OK %TIME% >> "%LOG%"
endlocal

REM ------------------------------------------------------- THE TWO ARMS, serialized
REM ⚠ not `|| exit /b %ERRORLEVEL%` -- percent expansion happens at PARSE time, i.e. BEFORE the
REM call runs, so that idiom returns a stale code. `if errorlevel 1` tests at RUN time.
call :run_arm sp4321_nokd spa 0
if errorlevel 1 exit /b 21
call :run_arm sp4321_kd   spb 1
if errorlevel 1 exit /b 22
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0

REM ---- :run_arm <tag> <prefix> <kd 0/1> -----------------------------------------------------
:run_arm
setlocal
set TAG=%1
set PFX=%2
set KD=%3
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
REM the adopted speed stack (training only; cleared before eval)
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
if "%KD%"=="1" (
  set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
  set RWKV_KD_ALPHA=0.5
) else (
  set RWKV_KD_MIX=
  set RWKV_KD_ALPHA=
)

echo === ARM %TAG% WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/seedpair65k/%PFX%_ws.toml > "%DIR%\%PFX%_ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%TAG%_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal
  exit /b 21
)
echo ARM %TAG% WS OK %TIME% >> "%LOG%"

REM KD is WS-only (iter 32 precedent); decay + eval run without it.
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

echo === ARM %TAG% DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/seedpair65k %PFX%_ws %PFX%_d scratchpad/seedpair65k/%PFX%_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\%PFX%_dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%TAG%_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/seedpair65k/%PFX%_decay.toml > "%DIR%\%PFX%_decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%TAG%_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal
  exit /b 23
)
echo ARM %TAG% DECAY OK %TIME% >> "%LOG%"

echo === ARM %TAG% EVAL TOML %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/seedpair65k %PFX%_d scratchpad/seedpair65k/%PFX%_eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 7500 > "%DIR%\%PFX%_etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%TAG%_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal
  exit /b 24
)
REM eval without the training speed flags, rectified, unsharded (d=80 rule)
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_EVAL_PAVA=1
echo === ARM %TAG% EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/seedpair65k/%PFX%_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%PFX%_eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% EVAL attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/seedpair65k/%PFX%_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%PFX%_eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% EVAL attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/seedpair65k/%PFX%_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%PFX%_eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%TAG%_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal
  exit /b 25
)
echo ARM %TAG% EVAL OK %TIME% >> "%LOG%"
endlocal
exit /b 0
