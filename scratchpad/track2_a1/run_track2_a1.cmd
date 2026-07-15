@echo off
REM ============================================================================
REM TRACK 2 A1 (first ablation): ALL channel mixers cut to cmf=1.0 -> 2,320,516
REM params (A0 2,762,884; cut 442,368). Exact A0 recipe otherwise (1 ep WS +
REM 0.25 ep decay, seed 1234, MAX=32768). GATE: 100k*dLL/dparams <= 0.0001 BOTH
REM modes vs A0 on the finite-user intersection => allowed degradation
REM <= 0.000442/mode. vprune vs A0's val trace (champion_5k_track2.json).
REM Eval = ONE process (d=128 unshardable). ~11 h. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a1\track2_a1.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
REM Clear EVERY step over the WHOLE run (WINDOW=0). Launch 4 (guard = first 1000 steps
REM only) crept 3.6->11.3 GB by step ~4100 (allocator envelope over variable d=128 group
REM shapes) -> WDDM paging, 4.3 s/step. Launch 5 (every=50) saturated 11.9/12 GB by step
REM ~250. Per-step clears hold 3.6 GB at an unchanged ~1.06 s/step (the ~0.1 s clear
REM hides under the ~1 s d=128 step; launch-4 steps 1-1000 prove it). Numerics-neutral.
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a1/architecture_d128_cmix1.py
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_a1/track2_a1_ws_trace.jsonl

echo ===== TRACK2_A1 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a1\track2_a1_ws_trace.jsonl scratchpad\track2_a1\track2_a1_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, d=128 cmix1.0 PLAIN, MAX=32768, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a1/track2_a1_ws.toml >> "%LOG%" 2>&1
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
REM MAX 32768 as the 10th arg (added 2026-07-15): write_decay_setup's default 110000 THRASHED
REM the d=128 decay (WDDM spill, ~1 step/min). Decay MAX must match the WS MAX.
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a1 t2a1ws t2a1d scratchpad/track2_a1/track2_a1_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a1/track2_a1_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a1.jsonl result\RWKV-P-track2_a1.jsonl result\RWKV-track2_a1-s0.jsonl result\RWKV-P-track2_a1-s0.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a1 t2a1d scratchpad/track2_a1/track2_a1_eval.toml RWKV-track2_a1 RWKV-P-track2_a1 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128 unshardable) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a1/track2_a1_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE INPUT: paired vs A0 on the finite-user intersection %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a1.jsonl --cand-imm result/RWKV-P-track2_a1.jsonl --champ-ahead result/RWKV-track2_a0.jsonl --champ-imm result/RWKV-P-track2_a0.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%; RATIO gate = 100k*dLL/442368 <= 0.0001 both modes, judged at record time) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
