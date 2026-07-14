@echo off
REM ============================================================================
REM RESEARCH ITER 15 (Andrew's directive 2026-07-14): drop the Anki review-state
REM input feature (scaled_state = dim 22: Filtered/Review/Learn/Relearn) via
REM RWKV_ZERO_FEATURES=22. Exact champ5k_plain recipe otherwise. ACCEPTANCE IS
REM DIRECTED: accept regardless of logloss delta (expected ~none) -- the final
REM paired_pvalue vs champ5k_plain is INFORMATIONAL. No vprune (must complete).
REM Launch DETACHED (detach.ps1) AFTER track2_a0 releases the GPU.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter15_nostate\iter15_nostate.log
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
set RWKV_STEP_TRACE=scratchpad/iter15_nostate/iter15_nostate_ws_trace.jsonl

echo ===== ITER15_NOSTATE START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter15_nostate\iter15_nostate_ws_trace.jsonl scratchpad\iter15_nostate\iter15_nostate_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, PLAIN, ZERO_FEATURES=22, no vprune) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter15_nostate/iter15_nostate_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.25 ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter15_nostate iter15ws iter15d scratchpad/iter15_nostate/iter15_nostate_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter15_nostate/iter15_nostate_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter15_nostate.jsonl result\RWKV-P-iter15_nostate.jsonl result\RWKV-iter15_nostate-solo.jsonl result\RWKV-P-iter15_nostate-solo.jsonl result\RWKV-iter15_nostate-s0.jsonl result\RWKV-P-iter15_nostate-s0.jsonl result\RWKV-iter15_nostate-s1.jsonl result\RWKV-P-iter15_nostate-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter15_nostate iter15d scratchpad/iter15_nostate/iter15_nostate_eval.toml RWKV-iter15_nostate RWKV-P-iter15_nostate 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (power-user-aware phased; RWKV_ZERO_FEATURES=22 active) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter15_nostate/iter15_nostate_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === INFO: paired vs champ5k_plain (directed accept -- delta is informational) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter15_nostate.jsonl --cand-imm result/RWKV-P-iter15_nostate.jsonl --champ-ahead result/RWKV-champ5k_plain.jsonl --champ-imm result/RWKV-P-champ5k_plain.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (informational paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
