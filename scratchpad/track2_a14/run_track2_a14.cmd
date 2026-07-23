@echo off
REM ============================================================================
REM TRACK-2 A14: LoRA-dim halving (decay/a/gate 16->8, v0-mix 8->4, ALL streams)
REM on the A13 champion recipe -- the first STRUCTURAL cut after the depth ladder
REM closed (A12: all floors mapped card2/deck4/note1/preset3/user3). The LoRA
REM projections are a distributed ~6% param mass that per-unit saliency cannot
REM target. 1,468,724 -> 1,380,660 params (-88,064 = -6.0% vs A13, -50.0% vs the
REM original 2.76M). Arch scratchpad/track2_a14/architecture_d128_lora8.py; strips
REM unchanged (9 entries). Gate vs A13 (0.298837/0.267805 val half): ratio
REM 100k*dLL/dparams <= 0.0001 BOTH modes on VAL 5001-7500 (allowed 0.000088/mode),
REM --intersect. VPRUNE_MIN_STEP=6000. Launch DETACHED via detach.ps1 (CRLF!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a14\track2_a14.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a14/architecture_d128_lora8.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_a14/track2_a14_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a14/t2a14_grad_stats_ws.json

echo ===== track2_a14 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a14\track2_a14_ws_trace.jsonl scratchpad\track2_a14\track2_a14_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, A13 recipe + LoRA dims halved, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a14/track2_a14_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a14/t2a14_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a14 t2a14ws t2a14d scratchpad/track2_a14/track2_a14_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a14/track2_a14_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a14.jsonl result\RWKV-P-track2_a14.jsonl result\RWKV-track2_a14-s0.jsonl result\RWKV-P-track2_a14-s0.jsonl result\RWKV-track2_a14.nanskip.jsonl result\RWKV-track2_a14-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a14 t2a14d scratchpad/track2_a14/track2_a14_eval.toml RWKV-track2_a14 RWKV-P-track2_a14 5001 7500 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a14/track2_a14_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs A13 champion (val half; ratio gate judged at record time) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a14.jsonl --cand-imm result/RWKV-P-track2_a14.jsonl --champ-ahead result/RWKV-track2_a13.jsonl --champ-imm result/RWKV-P-track2_a13.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
