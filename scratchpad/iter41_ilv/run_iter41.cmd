@echo off
REM ===========================================================================================
REM ITER 41: RWKV_INTERLEAVE=1 + TRUE FINE-TO-COARSE ORDER (2026-08-09, family: TOPOLOGY, NEW
REM -- Andrew's "try something more ambitious" directive, then his order directive after the
REM audit: every historical arch file ran card->DECK->NOTE; this run uses the corrected
REM card->NOTE->DECK->preset->user via architecture_d80_lora4_cnd.py. A 2-change bundle BY
REM DIRECTION (Andrew 2026-08-09) -- under interleaving the within-round order matters less
REM (every scope hears every other across rounds), so the interleave is the main effect.
REM Smoke re-run under the reordered arch: depth-1 oracle BIT-EXACT, order banner
REM [2,1,4,3,3] = card,note,deck,preset,user, no-grad sets identical (67 design-dead).
REM The RNN deploy path routes states BY NAME since the iter-41 refactor (golden-exact).
REM WHY: the sequential form gives card->deck->note->preset->user exactly ONE pass, so global
REM context can never influence card-level processing. Interleaving round-robins the SAME
REM layers across scopes (round r = layer r of every stream that has one, hierarchy order
REM within each round): same params, same per-entity states, same per-layer ops -- only the
REM execution ORDER changes, and from round 1 the specific streams see the general streams'
REM output. Depths [2,4,1,3,3] -> 4 rounds, 13 layer-steps.
REM VERIFIED BEFORE LAUNCH (scratchpad/parity3/smoke_interleave.py): the depth-1 oracle is
REM BIT-EXACT vs the sequential branch (gather composition correct), real depths differ,
REM no-grad sets identical, scripted compile OK. The RNN mirror reads the same flag; the
REM checkpoint-level trace parity runs before any interleaved champion ships.
REM RECIPE = the iter-39 champion exactly (seed 4321, KD alpha 0.9 WS-only, PAVA lambda 0.2,
REM tuned HPs). Gate pairs vs RWKV-iter39_kda09-s0.jsonl, RECTIFIED eval.
REM ⚠ WAITS for iter 40's chain to free the GPU first (anchored findstr -- the waitloop trap).
REM ⚠ NO del of result jsonls (fresh tags; retries resume from banked users).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter41_ilv
set LOG=%DIR%\iter41.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== ITER 41 (interleave) WAITING for iter 40 %DATE% %TIME% ===== > "%LOG%"
:waitloop
findstr /B /C:"DONE_EXIT_" "C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter40_kda1\iter40.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 300 /nobreak >nul
  goto waitloop
)
echo ===== ITER 41 (interleave) START %DATE% %TIME% ===== >> "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
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
REM ---- THE LEVER ----
set RWKV_INTERLEAVE=1
REM the adopted speed stack (training only; cleared before eval)
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
REM KD from the seed-pair dump (WS-only; cleared before decay), champion alpha
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.9

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter41_ilv/i41_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
findstr /C:"[interleave] round-robin layer schedule ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOILV %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
echo WS OK %TIME% >> "%LOG%"

set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_GRAD_STATS=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter41_ilv i41_ws i41_d scratchpad/iter41_ilv/i41_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter41_ilv/i41_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter41_ilv i41_d scratchpad/iter41_ilv/i41_eval.toml RWKV-iter41_ilv RWKV-P-iter41_ilv 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
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
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter41_ilv/i41_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter41_ilv/i41_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter41_ilv/i41_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
