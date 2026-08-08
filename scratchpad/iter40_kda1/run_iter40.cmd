@echo off
REM ===========================================================================================
REM ITER 40: RWKV_KD_ALPHA -> 1.0 (2026-08-08, family: distillation; the dose-curve endpoint).
REM WHY: alpha=0.5 was iter 32's FIRST GUESS and was never tuned; the d=128 teacher (12-ep
REM At 1.0 the WS phase trains purely on teacher soft targets; ground truth returns in the
REM decay phase (a pretrain/finetune split). Worse => the peak is bracketed and 0.9 stands.
REM the t128_seedpair_65k dump is reused free (alpha is applied at loss time, not dump time).
REM RECIPE = the iter-36 champion exactly (seed 4321, PAVA lambda 0.2, tuned HPs). KD WS-only.
REM Gate pairs vs RWKV-iter36_pava02-s0.jsonl, RECTIFIED eval.
REM ⚠ NO del of result jsonls (fresh tags; retries resume from banked users).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter40_kda1
set LOG=%DIR%\iter40.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== ITER 40 (kd alpha 1.0) START %DATE% %TIME% ===== > "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
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
REM KD from the seed-pair dump (WS-only; cleared before decay)
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
REM ---- THE LEVER (iter 32/35/36/37 = 0.5) ----
set RWKV_KD_ALPHA=1.0

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter40_kda1/i40_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
findstr /C:"alpha=1.0000" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOALPHA %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
echo WS OK %TIME% >> "%LOG%"

set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_GRAD_STATS=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter40_kda1 i40_ws i40_d scratchpad/iter40_kda1/i40_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter40_kda1/i40_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
echo DECAY OK %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter40_kda1 i40_d scratchpad/iter40_kda1/i40_eval.toml RWKV-iter40_kda10 RWKV-P-iter40_kda10 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM eval without the training speed flags, rectified, unsharded (d=80 rule)
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter40_kda1/i40_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter40_kda1/i40_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter40_kda1/i40_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
