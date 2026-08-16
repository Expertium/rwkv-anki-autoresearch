@echo off
REM ===========================================================================================
REM ITER 48 PHASE 2: EVAL ONLY. The chain died at DONE_EXIT_EVALFAIL_1 on a TorchScript runtime
REM bug in take_rank1_penalty (unannotated jit.ignore returning None == undefined tensor), fixed
REM 2026-08-15 and verified by a one-user scripted eval. WS and DECAY are intact on disk, so only
REM the eval is re-run -- a phase-2 runner, per the iters 43/46 precedent, because editing a .cmd
REM is unsafe and this one has already exited anyway.
REM
REM RWKV_RCOUPLE stays SET: it is a MODEL property, and rcouple_w is a conditional Parameter, so
REM clearing it would both score a different function and fail the strict checkpoint load.
REM RWKV_NO_JIT stays UNSET: the eval scripts the model, which is the path the bug lived on.
REM NO del of result jsonls -- eval_sharded resumes from banked users.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter48_rcouple
set LOG=%DIR%\iter48_eval.log
set STAMP=%RANDOM%%RANDOM%

echo ===== ITER 48 PHASE2 EVAL START %DATE% %TIME% ===== > "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_RCOUPLE=1
set RWKV_EVAL_PAVA=1

echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter48_rcouple/i48_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\p2eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter48_rcouple/i48_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\p2eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
