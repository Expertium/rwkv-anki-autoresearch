@echo off
REM ============================================================================
REM TRACK 2 A2: DECK stream 4 -> 3 layers on the A1 champion arch (all mixers
REM 1.0, 2,320,516 params; expected cut ~110k). Exact A1 recipe otherwise (1 ep
REM WS + 0.25 ep decay, seed 1234, MAX=32768). GATE: 100k*dLL/dparams <= 0.0001
REM BOTH modes vs A1 (full n=5000 -- A1 has 0 nanskips; --intersect self-heals
REM if A2 skips). vprune vs A1's val trace (champion_5k_track2.json).
REM RECORDS RWKV_GRAD_STATS (Andrew 2026-07-16) -> t2a2_grad_stats_{ws,decay}.json.
REM Eval = ONE process (d=128 unshardable). ~11 h. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a2\track2_a2.log
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
set RWKV_ARCH_MODULE=scratchpad/track2_a2/architecture_d128_cmix1_deck3.py
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_a2/track2_a2_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a2/t2a2_grad_stats_ws.json

echo ===== TRACK2_A2 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a2\track2_a2_ws_trace.jsonl scratchpad\track2_a2\track2_a2_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, d=128 cmix1.0 PLAIN, MAX=32768, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a2/track2_a2_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a2/t2a2_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep) %TIME% === >> "%LOG%"
REM MAX 32768 as the 10th arg (added 2026-07-15): write_decay_setup's default 110000 THRASHED
REM the d=128 decay (WDDM spill, ~1 step/min). Decay MAX must match the WS MAX.
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a2 t2a2ws t2a2d scratchpad/track2_a2/track2_a2_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a2/track2_a2_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a2.jsonl result\RWKV-P-track2_a2.jsonl result\RWKV-track2_a2-s0.jsonl result\RWKV-P-track2_a2-s0.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a2 t2a2d scratchpad/track2_a2/track2_a2_eval.toml RWKV-track2_a2 RWKV-P-track2_a2 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128 unshardable) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a2/track2_a2_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE INPUT: paired vs A1 (full n=5000 expected) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a2.jsonl --cand-imm result/RWKV-P-track2_a2.jsonl --champ-ahead result/RWKV-track2_a1.jsonl --champ-imm result/RWKV-P-track2_a1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%; RATIO gate = 100k*dLL/dparams <= 0.0001 both modes vs A1, judged at record time) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
