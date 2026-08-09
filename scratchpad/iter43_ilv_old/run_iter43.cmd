@echo off
REM ===========================================================================================
REM ITER 43: INTERLEAVE + THE ORIGINAL STREAM ORDER (2026-08-09, family TOPOLOGY).
REM THE MISSING 4th CELL of the (order) x (schedule) 2x2 that iter 42 completed three of:
REM   old order + sequential  = iter 39: 0.298180 / 0.265875
REM   new order + sequential  = iter 42: 0.298379 / 0.266090   (order alone is NEGATIVE)
REM   new order + interleaved = iter 41: 0.297889 / 0.265479   (champion)
REM   old order + interleaved = THIS RUN
REM iter 42 showed the reorder costs -0.000198/-0.000215 sequentially while interleaving is
REM worth +0.000489/+0.000612. If the order penalty is ~additive under interleaving this lands
REM ~0.29769/0.26526 and PASSES the gate vs iter 41; if it vanishes, that is itself the finding
REM (within-round order stops mattering once every scope hears every other across rounds).
REM LEVER = RWKV_ARCH_MODULE -> the ORIGINAL architecture_d80_lora4.py (card->deck->note->
REM preset->user, depths 2,4,1,3,3). Everything else = the iter-41 champion env.
REM Guards: the ws log MUST carry the interleave banner AND MUST NOT name the _cnd arch (a
REM copied env line would silently re-run iter 41).
REM ⚠ NO del of result jsonls (fresh tags; retries resume from banked users).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter43_ilv_old
set LOG=%DIR%\iter43.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== ITER 43 (interleave + original order) START %DATE% %TIME% ===== > "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
REM ---- THE LEVER: the ORIGINAL order, not the _cnd reorder ----
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
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
set RWKV_GRAD_STATS=%DIR%\grad_stats.json
REM the adopted speed stack (training only; cleared before eval)
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
REM KD from the seed-pair dump (WS-only; cleared before decay), champion alpha
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.9

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter43_ilv_old/i43_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
findstr /C:"[interleave] round-robin layer schedule ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOILV %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
findstr /C:"architecture_d80_lora4_cnd.py" "%DIR%\ws_%STAMP%.log" >nul
if %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGARCH %DATE% %TIME% >> "%LOG%"
  exit /b 30
)
echo WS OK %TIME% >> "%LOG%"

set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_GRAD_STATS=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter43_ilv_old i43_ws i43_d scratchpad/iter43_ilv_old/i43_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter43_ilv_old/i43_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
findstr /C:"[interleave] round-robin layer schedule ON" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOILV_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 29
)
echo DECAY OK %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter43_ilv_old i43_d scratchpad/iter43_ilv_old/i43_eval.toml RWKV-iter43_ilvold RWKV-P-iter43_ilvold 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM eval without the training speed flags, rectified, unsharded (d=80 rule).
REM ⚠ RWKV_INTERLEAVE stays SET -- eval must score the interleaved model.
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter43_ilv_old/i43_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter43_ilv_old/i43_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter43_ilv_old/i43_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
