@echo off
REM ============================================================================
REM RESEARCH ITER 20 (2026-07-16): CROSS-HEAD READOUT MIX (RWKV_XHEAD_MIX=1,
REM zero-init delta before out_group_norm, +896 params). Exact iter-15 recipe otherwise
REM (incl. RWKV_ZERO_FEATURES=22). Normal gated candidate: vprune vs the iter-15
REM champion's val trace; accept = >=0.0003 BOTH + p<1e-4 BOTH at full eval.
REM Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter20_xmix\iter20_xmix.log
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
set RWKV_XHEAD_MIX=1
set RWKV_STEP_TRACE=scratchpad/iter20_xmix/iter20_xmix_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER19_PBIN025 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter20_xmix\iter20_xmix_ws_trace.jsonl scratchpad\iter20_xmix\iter20_xmix_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, PLAIN, XHEAD_MIX + ZERO_FEATURES=22, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter20_xmix/iter20_xmix_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter20_xmix iter20ws iter20d scratchpad/iter20_xmix/iter20_xmix_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter20_xmix/iter20_xmix_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter20_xmix.jsonl result\RWKV-P-iter20_xmix.jsonl result\RWKV-iter20_xmix-solo.jsonl result\RWKV-P-iter20_xmix-solo.jsonl result\RWKV-iter20_xmix-s0.jsonl result\RWKV-P-iter20_xmix-s0.jsonl result\RWKV-iter20_xmix-s1.jsonl result\RWKV-P-iter20_xmix-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter20_xmix iter20d scratchpad/iter20_xmix/iter20_xmix_eval.toml RWKV-iter20_xmix RWKV-P-iter20_xmix 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (power-user-aware phased; XHEAD_MIX + ZERO_FEATURES active) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter20_xmix/iter20_xmix_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter15 champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter20_xmix.jsonl --cand-imm result/RWKV-P-iter20_xmix.jsonl --champ-ahead result/RWKV-iter15_nostate.jsonl --champ-imm result/RWKV-P-iter15_nostate.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
