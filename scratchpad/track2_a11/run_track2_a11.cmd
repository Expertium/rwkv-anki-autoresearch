@echo off
REM ============================================================================
REM TRACK-2 A11: the A10 bundle MINUS the note_id:0 strip -- user 3L->2L (same
REM arch module as A10) + deck_id:3 mixer strip; note.L0 mixer KEPT. 1,468,724 ->
REM 1,352,620 params (-116,104 = -7.9% vs A9). De-bundles A10's reject: PASS ->
REM banks the size + fingers note.L0 as the poison; FAIL -> user depth floors at
REM 3L. Gate vs A9 (0.298625/0.267615 val half): ratio <= 0.0001 BOTH modes on
REM VAL 5001-7500 (allowed 0.000116/mode), --intersect. VPRUNE_MIN_STEP=6000.
REM Launch DETACHED via detach.ps1 (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a11\track2_a11.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a10/architecture_d128_cmix1_user2_card2_note1.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,deck_id:3,card_id:1
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_a11/track2_a11_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a11/t2a11_grad_stats_ws.json

echo ===== track2_a11 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a11\track2_a11_ws_trace.jsonl scratchpad\track2_a11\track2_a11_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, d=128 A9 recipe + user2L + deck3 cmix strip, note.L0 mixer kept, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a11/track2_a11_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a11/t2a11_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a11 t2a11ws t2a11d scratchpad/track2_a11/track2_a11_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a11/track2_a11_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a11.jsonl result\RWKV-P-track2_a11.jsonl result\RWKV-track2_a11-s0.jsonl result\RWKV-P-track2_a11-s0.jsonl result\RWKV-track2_a11.nanskip.jsonl result\RWKV-track2_a11-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a11 t2a11d scratchpad/track2_a11/track2_a11_eval.toml RWKV-track2_a11 RWKV-P-track2_a11 5001 7500 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a11/track2_a11_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs A9 champion (val half; ratio gate judged at record time) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a11.jsonl --cand-imm result/RWKV-P-track2_a11.jsonl --champ-ahead result/RWKV-track2_a9.jsonl --champ-imm result/RWKV-P-track2_a9.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
