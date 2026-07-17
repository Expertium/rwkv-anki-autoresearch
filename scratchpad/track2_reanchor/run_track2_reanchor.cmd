@echo off
REM ============================================================================
REM TRACK-2 NO-RESIDUAL RE-ANCHOR: A1 arch + RWKV_NO_AHEAD_RESIDUAL=1.
REM Re-anchors the track-2 reference for the mandatory no-residual recipe (as
REM iter 22 did for track 1); A3 re-gates vs this run. Waits for iter 23's
REM DONE_EXIT (STEP 0), then ~11 h. RWKV_GRAD_STATS on (fixed recorder).
REM Andrew can veto: kill this cmd's pid tree before iter 23 finishes.
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_reanchor\track2_reanchor.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a1/architecture_d128_cmix1.py
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_reanchor/track2_reanchor_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_reanchor/t2re_grad_stats_ws.json

echo ===== TRACK2_REANCHOR START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_reanchor\track2_reanchor_ws_trace.jsonl scratchpad\track2_reanchor\track2_reanchor_ws_trace.jsonl.val.jsonl 2>nul
echo === STEP 0: wait for iter 23 to release the GPU %TIME% === >> "%LOG%"
:waitloop
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter23_pava\iter23_pava.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop
)
echo iter 23 done -- starting re-anchor %TIME% >> "%LOG%"

echo === WS 1 epoch (1-5000, d=128 A1 arch NO_AHEAD_RESIDUAL, MAX=32768, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_reanchor/track2_reanchor_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_reanchor/t2re_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_reanchor t2rews t2red scratchpad/track2_reanchor/track2_reanchor_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_reanchor/track2_reanchor_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_reanchor.jsonl result\RWKV-P-track2_reanchor.jsonl result\RWKV-track2_reanchor-s0.jsonl result\RWKV-P-track2_reanchor-s0.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_reanchor t2red scratchpad/track2_reanchor/track2_reanchor_eval.toml RWKV-track2_reanchor RWKV-P-track2_reanchor 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128 unshardable) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_reanchor/track2_reanchor_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === INFO: paired vs A1 (residual-removal cost at d=128) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_reanchor.jsonl --cand-imm result/RWKV-P-track2_reanchor.jsonl --champ-ahead result/RWKV-track2_a1.jsonl --champ-imm result/RWKV-P-track2_a1.jsonl >> "%LOG%" 2>&1
echo === INFO: paired A3 vs THIS anchor (the deferred A3 verdict input) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a3.jsonl --cand-imm result/RWKV-P-track2_a3.jsonl --champ-ahead result/RWKV-track2_reanchor.jsonl --champ-imm result/RWKV-P-track2_reanchor.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (both paired outputs above; ratio gate judged at record time) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
