@echo off
REM ============================================================================
REM TRACK-2 A16: d_model 96->64 (N_HEADS 3->2, K=32 kept) -- THE GOAL-LINE CUT
REM (delegated by Andrew 2026-07-23; the >=5x-reduction path). Depth/strips/recipe
REM unchanged from the champion; LoRA dims at the A14 halving. VERIFIED at launch
REM 2026-07-25: 388,032 params = -52.0% vs A15 (808,762) = 7.11x below the
REM original 2.76M -- PAST Andrews >=5x goal; per-card state 4,352 vs 6,528 (-33%).
REM GATE: ratio vs A15 on VAL 5001-7500 -- dparams 420,730 => allowed +0.000421
REM per mode (both modes must stay inside; shrinking state is welcome, not gated).
REM LOG HYGIENE (2026-07-24 lesson): the control log is written ONLY by this cmd;
REM every python phase redirects to its own STAMPED sublog, so a straggler worker
REM inheriting a phase's handle cannot block the cmd's >> and silently skip phases.
REM VPRUNE_MIN_STEP=6000, ref = champion_5k_track2.json (= A15). Launch DETACHED
REM via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a16
set LOG=%DIR%\track2_a16.log
set STAMP=%RANDOM%%RANDOM%
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a16/architecture_d64_lora8.py
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
set RWKV_STEP_TRACE=scratchpad/track2_a16/track2_a16_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a16/t2a16_grad_stats_ws.json

echo ===== track2_a16 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a16\track2_a16_ws_trace.jsonl scratchpad\track2_a16\track2_a16_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, champion recipe + d_model 96, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a16/track2_a16_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
echo WS OK %TIME% >> "%LOG%"
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a16/t2a16_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a16 t2a16ws t2a16d scratchpad/track2_a16/track2_a16_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 > "%DIR%\decay_setup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a16/track2_a16_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo DECAY OK %TIME% >> "%LOG%"

del /Q result\RWKV-track2_a16.jsonl result\RWKV-P-track2_a16.jsonl result\RWKV-track2_a16-s0.jsonl result\RWKV-P-track2_a16-s0.jsonl result\RWKV-track2_a16.nanskip.jsonl result\RWKV-track2_a16-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a16 t2a16d scratchpad/track2_a16/track2_a16_eval.toml RWKV-track2_a16 RWKV-P-track2_a16 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
echo === EVAL (single process, d=64, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a16/track2_a16_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === GATE: paired vs CHAMPION A15 (val half; ratio gate judged at record time) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a16.jsonl --cand-imm result/RWKV-P-track2_a16.jsonl --champ-ahead result/RWKV-track2_a15.jsonl --champ-imm result/RWKV-P-track2_a15.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
