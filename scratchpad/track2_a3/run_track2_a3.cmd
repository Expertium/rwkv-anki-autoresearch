@echo off
REM ============================================================================
REM TRACK 2 A3: GRU-faithful curve head (RWKV_GRU_HEAD=2, N=2 predicted w/S/decay
REM power curves; srs-benchmark models/gru.py class GRU -- NOT "GRU-P", that name
REM was removed upstream) + dead ahead head stripped to 1x1 dummies.
REM Params: 2,126,224 on A1 arch / 2,010,120 on A2 deck3 arch.
REM *** AT LAUNCH: set RWKV_ARCH_MODULE + the paired_pvalue champ files below to
REM *** match A2's verdict (accepted -> deck3 arch + A2 results as champion).
REM Exact A1/A2 recipe: 1 ep WS + 0.25 ep decay, seed 1234, MAX=32768.
REM vprune vs champion_5k_track2.json (auto-updated if A2 promoted).
REM RWKV_GRAD_STATS mandatory (A2+ directive). Eval = ONE process (d=128).
REM ~11 h. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a3\track2_a3.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
REM per-step cache clears over the whole run (track-2 rule; see A2 cmd for the history)
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
REM *** ARCH: A1 cmix1 default; switch to track2_a2/architecture_d128_cmix1_deck3.py if A2 accepted
set RWKV_ARCH_MODULE=scratchpad/track2_a1/architecture_d128_cmix1.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_a3/track2_a3_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a3/t2a3_grad_stats_ws.json

echo ===== TRACK2_A3 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a3\track2_a3_ws_trace.jsonl scratchpad\track2_a3\track2_a3_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, d=128 GRU head N=2, MAX=32768, vprune ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a3/track2_a3_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a3/t2a3_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a3 t2a3ws t2a3d scratchpad/track2_a3/track2_a3_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a3/track2_a3_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a3.jsonl result\RWKV-P-track2_a3.jsonl result\RWKV-track2_a3-s0.jsonl result\RWKV-P-track2_a3-s0.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a3 t2a3d scratchpad/track2_a3/track2_a3_eval.toml RWKV-track2_a3 RWKV-P-track2_a3 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128 unshardable) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a3/track2_a3_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE INPUT: paired vs the track-2 champion %TIME% === >> "%LOG%"
REM *** AT LAUNCH: point champ files at A2's results if A2 was accepted
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a3.jsonl --cand-imm result/RWKV-P-track2_a3.jsonl --champ-ahead result/RWKV-track2_a1.jsonl --champ-imm result/RWKV-P-track2_a1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%; RATIO gate = 100k*dLL/dparams both modes, judged at record time; re-anchor decision = Andrew) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
