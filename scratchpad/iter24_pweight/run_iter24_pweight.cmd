@echo off
REM ============================================================================
REM RESEARCH ITER 24: iter 23 + p-head BUTTON-PROBABILITY pooling weights
REM (RWKV_PAVA_PWEIGHT=1 -- weighted power mean, weights = Instant-mode softmax
REM at the paired query row; Andrew's fixed queue iter 23 -> 24). Otherwise the
REM exact iter-23 config. Gate: >=0.0003 BOTH modes vs the current champion +
REM p<0.0001. Waits for the track-2 re-anchor's DONE_EXIT (STEP 0).
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM NOTE: review lambda/density against iter 23's verdict BEFORE launching.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter24_pweight\iter24_pweight.log
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
set RWKV_PAVA_PWEIGHT=1
set RWKV_PROBE_DENSITY=0.08
set RWKV_STEP_TRACE=scratchpad/iter24_pweight/iter24_pweight_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER24_PWEIGHT START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for the track-2 re-anchor to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_reanchor\track2_reanchor.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo re-anchor done -- starting iter 24 %TIME% >> "%LOG%"
del /Q scratchpad\iter24_pweight\iter24_pweight_ws_trace.jsonl scratchpad\iter24_pweight\iter24_pweight_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, PAVA lambda=0.1 density=0.08, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter24_pweight/iter24_pweight_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter24_pweight iter24ws iter24d scratchpad/iter24_pweight/iter24_pweight_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter24_pweight/iter24_pweight_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter24_pweight.jsonl result\RWKV-P-iter24_pweight.jsonl result\RWKV-iter24_pweight-solo.jsonl result\RWKV-P-iter24_pweight-solo.jsonl result\RWKV-iter24_pweight-s0.jsonl result\RWKV-P-iter24_pweight-s0.jsonl result\RWKV-iter24_pweight-s1.jsonl result\RWKV-P-iter24_pweight-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter24_pweight iter24d scratchpad/iter24_pweight/iter24_pweight_eval.toml RWKV-iter24_pweight RWKV-P-iter24_pweight 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (phased sharded; probes OFF in eval by construction) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter24_pweight/iter24_pweight_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter22 champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter24_pweight.jsonl --cand-imm result/RWKV-P-iter24_pweight.jsonl --champ-ahead result/RWKV-iter22_nores.jsonl --champ-imm result/RWKV-P-iter22_nores.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
