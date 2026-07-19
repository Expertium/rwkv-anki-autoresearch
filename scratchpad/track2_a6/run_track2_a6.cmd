@echo off
REM ============================================================================
REM TRACK-2 A6: channel-mixer thinning bundle on the A5 champion recipe.
REM RWKV_STRIP_CMIX removes the 5 stable bottom-saliency mixers (deck.L1,
REM preset.L1, preset.L2, user.L1, user.L2 -- consistent across 3 grad-stats
REM recordings). 2,115,359 -> 1,949,624 params (-165,735 = -7.83% vs A5).
REM Gate vs A5 (0.300532/0.269127): ratio 100k*dLL/dparams <= 0.0001 BOTH modes,
REM full n=5000. VPRUNE_MIN_STEP=6000 (GRU zero-init prior). ~11 h.
REM STEP 0 waits for iter 27's DONE_EXIT. Launch DETACHED (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a6\track2_a6.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a1/architecture_d128_cmix1.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:1,user_id:2,preset_id:1,preset_id:2,deck_id:1
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_a6/track2_a6_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a6/t2a6_grad_stats_ws.json

echo ===== TRACK2_A6 START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for iter 27 to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter27_gru4\iter27_gru4.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo iter 27 done -- starting A6 %TIME% >> "%LOG%"
del /Q scratchpad\track2_a6\track2_a6_ws_trace.jsonl scratchpad\track2_a6\track2_a6_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, d=128 A5 recipe + cmix strip, MAX=32768, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a6/track2_a6_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a6/t2a6_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a6 t2a6ws t2a6d scratchpad/track2_a6/track2_a6_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a6/track2_a6_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a6.jsonl result\RWKV-P-track2_a6.jsonl result\RWKV-track2_a6-s0.jsonl result\RWKV-P-track2_a6-s0.jsonl result\RWKV-track2_a6.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a6 t2a6d scratchpad/track2_a6/track2_a6_eval.toml RWKV-track2_a6 RWKV-P-track2_a6 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a6/track2_a6_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs A5 champion (ratio gate judged at record time) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a6.jsonl --cand-imm result/RWKV-P-track2_a6.jsonl --champ-ahead result/RWKV-track2_a5.jsonl --champ-imm result/RWKV-P-track2_a5.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
