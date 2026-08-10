@echo off
REM ===========================================================================================
REM ITER 44: RWKV_ILV_SPREAD -- endpoint-anchored layer PLACEMENT inside the interleaved
REM schedule (2026-08-10, family TOPOLOGY). One flag on the iter-41 champion recipe.
REM
REM WHY: iter 43 completed the (order) x (schedule) 2x2 and its verdict was "the SCHEDULE is
REM the productive lever, the order is not" -- interleaving is worth +0.0002..0.0006 in both
REM modes regardless of stream order, while the order itself is null under interleaving. This
REM run pushes on the schedule, at a concrete deficiency of the front-loaded placement: layer j
REM runs in round j, so a SHALLOW stream only ever runs EARLY and can never consume the
REM cross-scope context that interleaving exists to expose. In the champion (_cnd depths
REM [2,1,4,3,3]) the NOTE stream has depth 1 -- it runs in round 0 and never again, feeding the
REM global context but never reading it.
REM SPREAD distributes each stream's layers across all rounds with the endpoints anchored
REM (layer 0 -> round 0, last layer -> last round; a depth-1 stream goes LAST):
REM   card [0,-,-,1]   note [-,-,-,0]   deck [0,1,2,3]   preset [0,-,1,2]   user [0,-,1,2]
REM so note now reads maximal context, and EVERY stream's final layer is computed in the last
REM round. Same params, same per-entity states, same op count -- placement only.
REM
REM PRE-VERIFIED (scratchpad/parity3/smoke_ilv_spread.py, all 4 checks):
REM   [1] spread OFF is BIT-IDENTICAL to a literal re-implementation of the old front-loaded
REM       loop (the schedule-lookup rewrite is inert when the flag is off);
REM   [2] depth-1 oracle: spread == front-loaded == sequential, bit-identical;
REM   [3] at real depths spread DIFFERS (not a null lever), grads finite, and the no-grad
REM       parameter set is IDENTICAL to front-loaded (placement strands no parameter);
REM   [4] compiles and runs SCRIPTED (JIT on = the configuration eval runs), checksum equal to
REM       eager. The RNN deploy mirror uses the SAME interleave_schedule() helper.
REM
REM RECIPE = the iter-41 champion exactly (seed 4321, KD alpha 0.9 WS-only, PAVA lambda 0.2,
REM tuned HPs, _cnd arch, RWKV_INTERLEAVE=1) PLUS RWKV_ILV_SPREAD=1. Gate vs iter 41.
REM Guards: the ws AND decay logs must show placement=SPREAD; a missing flag would silently
REM re-run the champion.
REM ⚠ DO NOT `git rebase`/`pull`/`checkout` anything while this runs -- iter 43's chain died
REM because a rebase rewrote its running .cmd and corrupted cmd.exe's saved read offset.
REM ⚠ NO del of result jsonls (fresh tags; retries resume from banked users).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter44_spread
set LOG=%DIR%\iter44.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== ITER 44 (ILV_SPREAD) START %DATE% %TIME% ===== > "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
REM ---- THE LEVER ----
set RWKV_ILV_SPREAD=1
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
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter44_spread/i44_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
findstr /C:"placement = SPREAD" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOSPREAD %DATE% %TIME% >> "%LOG%"
  exit /b 33
)
findstr /C:"architecture_d80_lora4_cnd.py" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGARCH %DATE% %TIME% >> "%LOG%"
  exit /b 30
)
echo WS OK %TIME% >> "%LOG%"

set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_GRAD_STATS=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter44_spread i44_ws i44_d scratchpad/iter44_spread/i44_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
findstr /C:"i44_ws_10935" "%DIR%\dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGCKPT %DATE% %TIME% >> "%LOG%"
  exit /b 32
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter44_spread/i44_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
findstr /C:"placement = SPREAD" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOSPREAD_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 34
)
echo DECAY OK %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter44_spread i44_d scratchpad/iter44_spread/i44_eval.toml RWKV-iter44_spread RWKV-P-iter44_spread 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM eval without the training speed flags, rectified, unsharded (d=80 rule).
REM ⚠ RWKV_INTERLEAVE + RWKV_ILV_SPREAD stay SET -- eval must score the model that was trained.
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter44_spread/i44_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter44_spread/i44_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter44_spread/i44_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
