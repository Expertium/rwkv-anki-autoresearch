@echo off
REM ============================================================================
REM RESEARCH ITER 22 REDEFINED (2026-07-16, Andrew's directive): DISABLE the piecewise-
REM linear curve correction (RWKV_NO_AHEAD_RESIDUAL=1 -> curve = pure mixture, monotone
REM by construction; supersedes the cummin variant which never ran; 193,724 params).
REM Iter-15 recipe (incl. RWKV_ZERO_FEATURES=22). Verdict = ANDREW DECIDES (report
REM finals + deltas vs iter15 + p-values + nan_users, then wait -- no auto-verdict).
REM Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter22_nores\iter22_nores.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_ZERO_FEATURES=22
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STEP_TRACE=scratchpad/iter22_nores/iter22_nores_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER22_NORES START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter22_nores\iter22_nores_ws_trace.jsonl scratchpad\iter22_nores\iter22_nores_ws_trace.jsonl.val.jsonl 2>nul
echo === STEP 0: wait for track-2 A2 to release the GPU %TIME% === >> "%LOG%"
:waitloop
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a2\track2_a2.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop
)
echo A2 done -- starting iter 22 %TIME% >> "%LOG%"

echo === WS 1 epoch (1-5000, PLAIN, NO_AHEAD_RESIDUAL + ZERO_FEATURES=22, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter22_nores/iter22_nores_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.25 ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter22_nores iter22nws iter22nd scratchpad/iter22_nores/iter22_nores_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter22_nores/iter22_nores_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter22_nores.jsonl result\RWKV-P-iter22_nores.jsonl result\RWKV-iter22_nores-solo.jsonl result\RWKV-P-iter22_nores-solo.jsonl result\RWKV-iter22_nores-s0.jsonl result\RWKV-P-iter22_nores-s0.jsonl result\RWKV-iter22_nores-s1.jsonl result\RWKV-P-iter22_nores-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter22_nores iter22nd scratchpad/iter22_nores/iter22_nores_eval.toml RWKV-iter22_nores RWKV-P-iter22_nores 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (power-user-aware phased; NO_AHEAD_RESIDUAL + ZERO_FEATURES active) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter22_nores/iter22_nores_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter15 champion (REPORT ONLY -- Andrew decides) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter22_nores.jsonl --cand-imm result/RWKV-P-iter22_nores.jsonl --champ-ahead result/RWKV-iter15_nostate.jsonl --champ-imm result/RWKV-P-iter15_nostate.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
