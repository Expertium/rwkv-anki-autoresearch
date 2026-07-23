@echo off
REM ============================================================================
REM TRACK-2 A15: d_model 128->96 (N_HEADS 4->3, K=32 kept) -- THE WIDTH CUT
REM (delegated by Andrew 2026-07-23; the >=5x-reduction path). Depth/strips/recipe
REM unchanged from the champion; LoRA dims at the A14 halving (flip the arch line
REM if A14 rejected). 808,762 params = -41.4% vs A14 (1,380,660), 3.42x below the
REM original 2.76M. Gate: ratio vs the CURRENT champion on VAL 5001-7500 --
REM vs A14 allowed 0.000572/mode (dparams 571,898); vs A13 allowed 0.000660/mode.
REM CONFIRM the paired_pvalue champ ref below matches the champion at launch time.
REM VPRUNE_MIN_STEP=6000. Launch DETACHED via detach.ps1 (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a15\track2_a15.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a15/architecture_d96_lora8.py
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
set RWKV_STEP_TRACE=scratchpad/track2_a15/track2_a15_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a15/t2a15_grad_stats_ws.json

echo ===== track2_a15 START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for baseline LSTM to release the GPU (chain: A14 - GRU - LSTM - A15) %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\baseline_lstm\baseline_lstm.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo LSTM baseline done -- starting A15 %TIME% >> "%LOG%"
del /Q scratchpad\track2_a15\track2_a15_ws_trace.jsonl scratchpad\track2_a15\track2_a15_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, champion recipe + d_model 96, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a15/track2_a15_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a15/t2a15_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a15 t2a15ws t2a15d scratchpad/track2_a15/track2_a15_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a15/track2_a15_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a15.jsonl result\RWKV-P-track2_a15.jsonl result\RWKV-track2_a15-s0.jsonl result\RWKV-P-track2_a15-s0.jsonl result\RWKV-track2_a15.nanskip.jsonl result\RWKV-track2_a15-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a15 t2a15d scratchpad/track2_a15/track2_a15_eval.toml RWKV-track2_a15 RWKV-P-track2_a15 5001 7500 >> "%LOG%" 2>&1
echo === EVAL (single process, d=96, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a15/track2_a15_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs CHAMPION (val half; ratio gate judged at record time) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a15.jsonl --cand-imm result/RWKV-P-track2_a15.jsonl --champ-ahead result/RWKV-track2_a14.jsonl --champ-imm result/RWKV-P-track2_a14.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
