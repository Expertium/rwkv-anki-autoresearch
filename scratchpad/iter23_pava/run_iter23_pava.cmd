@echo off
REM ============================================================================
REM RESEARCH ITER 23: learnable power-mean PAVA rectifier + counterfactual
REM button-probe rows (MONOTONICITY_PLAN.md stage 2). Champion iter-22 recipe
REM (NO_AHEAD_RESIDUAL, ZERO_FEATURES=22, H=2/K=16) + RWKV_PAVA_LAMBDA=0.1 +
REM RWKV_PROBE_DENSITY=0.08. Gate: >=0.0003 BOTH modes vs iter 22 + p<0.0001.
REM vprune vs champion_5k_plain.json (= iter 22's val trace).
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter23_pava\iter23_pava.log
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
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_STEP_TRACE=scratchpad/iter23_pava/iter23_pava_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER23_PAVA START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter23_pava\iter23_pava_ws_trace.jsonl scratchpad\iter23_pava\iter23_pava_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, PAVA lambda=0.1 density=0.08, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter23_pava/iter23_pava_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter23_pava iter23ws iter23d scratchpad/iter23_pava/iter23_pava_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter23_pava/iter23_pava_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter23_pava.jsonl result\RWKV-P-iter23_pava.jsonl result\RWKV-iter23_pava-solo.jsonl result\RWKV-P-iter23_pava-solo.jsonl result\RWKV-iter23_pava-s0.jsonl result\RWKV-P-iter23_pava-s0.jsonl result\RWKV-iter23_pava-s1.jsonl result\RWKV-P-iter23_pava-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter23_pava iter23d scratchpad/iter23_pava/iter23_pava_eval.toml RWKV-iter23_pava RWKV-P-iter23_pava 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (phased sharded; probes OFF in eval by construction) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter23_pava/iter23_pava_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter22 champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter23_pava.jsonl --cand-imm result/RWKV-P-iter23_pava.jsonl --champ-ahead result/RWKV-iter22_nores.jsonl --champ-imm result/RWKV-P-iter22_nores.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
