@echo off
REM ============================================================================
REM RESEARCH ITER 22 (2026-07-16): MONOTONE CURVES (RWKV_MONO_CURVES=1, cummin on the
REM ahead-logit residual -> curve non-increasing in t; 193,724 params). Iter-15 recipe
REM (incl. RWKV_ZERO_FEATURES=22). Normal gated candidate: vprune vs the iter-15
REM champion's val trace; accept = >=0.0003 BOTH + p<1e-4 BOTH at full eval.
REM Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter22_monocurve\iter22_monocurve.log
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
set RWKV_MONO_CURVES=1
set RWKV_STEP_TRACE=scratchpad/iter22_monocurve/iter22_monocurve_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER22_MONOCURVE START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter22_monocurve\iter22_monocurve_ws_trace.jsonl scratchpad\iter22_monocurve\iter22_monocurve_ws_trace.jsonl.val.jsonl 2>nul
echo === STEP 0: wait for track-2 A2 to release the GPU %TIME% === >> "%LOG%"
:waitloop
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a2\track2_a2.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop
)
echo A2 done -- starting iter 22 %TIME% >> "%LOG%"

echo === WS 1 epoch (1-5000, PLAIN, MONO_CURVES + ZERO_FEATURES=22, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter22_monocurve/iter22_monocurve_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter22_monocurve iter22ws iter22d scratchpad/iter22_monocurve/iter22_monocurve_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter22_monocurve/iter22_monocurve_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter22_monocurve.jsonl result\RWKV-P-iter22_monocurve.jsonl result\RWKV-iter22_monocurve-solo.jsonl result\RWKV-P-iter22_monocurve-solo.jsonl result\RWKV-iter22_monocurve-s0.jsonl result\RWKV-P-iter22_monocurve-s0.jsonl result\RWKV-iter22_monocurve-s1.jsonl result\RWKV-P-iter22_monocurve-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter22_monocurve iter22d scratchpad/iter22_monocurve/iter22_monocurve_eval.toml RWKV-iter22_monocurve RWKV-P-iter22_monocurve 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (power-user-aware phased; MONO_CURVES + ZERO_FEATURES active) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter22_monocurve/iter22_monocurve_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter15 champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter22_monocurve.jsonl --cand-imm result/RWKV-P-iter22_monocurve.jsonl --champ-ahead result/RWKV-iter15_nostate.jsonl --champ-imm result/RWKV-P-iter15_nostate.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
