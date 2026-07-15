@echo off
REM ============================================================================
REM RESEARCH ITER 16 (2026-07-15): PREHEAD OUTPUT GATE (RWKV_PREHEAD_GATE=1,
REM zero-init identity, +1,056 params). Exact iter-15 champion recipe otherwise
REM (incl. RWKV_ZERO_FEATURES=22). Normal gated candidate: vprune vs the iter-15
REM champion's val trace; accept = >=0.0003 BOTH + p<1e-4 BOTH at full eval.
REM Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter16_pregate\iter16_pregate.log
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
set RWKV_PREHEAD_GATE=1
set RWKV_STEP_TRACE=scratchpad/iter16_pregate/iter16_pregate_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER16_PREGATE START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter16_pregate\iter16_pregate_ws_trace.jsonl scratchpad\iter16_pregate\iter16_pregate_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, PLAIN, PREHEAD_GATE + ZERO_FEATURES=22, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter16_pregate/iter16_pregate_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter16_pregate iter16ws iter16d scratchpad/iter16_pregate/iter16_pregate_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter16_pregate/iter16_pregate_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter16_pregate.jsonl result\RWKV-P-iter16_pregate.jsonl result\RWKV-iter16_pregate-solo.jsonl result\RWKV-P-iter16_pregate-solo.jsonl result\RWKV-iter16_pregate-s0.jsonl result\RWKV-P-iter16_pregate-s0.jsonl result\RWKV-iter16_pregate-s1.jsonl result\RWKV-P-iter16_pregate-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter16_pregate iter16d scratchpad/iter16_pregate/iter16_pregate_eval.toml RWKV-iter16_pregate RWKV-P-iter16_pregate 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (power-user-aware phased; GATE + ZERO_FEATURES active) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter16_pregate/iter16_pregate_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter15 champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter16_pregate.jsonl --cand-imm result/RWKV-P-iter16_pregate.jsonl --champ-ahead result/RWKV-iter15_nostate.jsonl --champ-imm result/RWKV-P-iter15_nostate.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
