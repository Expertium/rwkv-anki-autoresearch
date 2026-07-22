@echo off
REM ============================================================================
REM TRACK-2 A13: PURE RE-ANCHOR (Andrew 2026-07-22) -- the A9 champion arch +
REM recipe with RWKV_ZERO_FEATURES=22 (Anki card-state input removed; track 1 has
REM had this since iter 15, track 2 never adopted it -- recipe divergence fix).
REM Params UNCHANGED 1,468,724 (arch = track2_a9's module). NO GATE: directed
REM re-baseline a la A4 -- promoted to track-2 reference at completion; the tail
REM paired_pvalue vs A9 is INFORMATIONAL (= the price/gain of state removal at
REM d=128; expected ~ +/-0.0001, "better without" at d=32 per iter 15).
REM All depth floors mapped (card2/deck4/note1/preset3/user3) -- structural cuts
REM (LoRA dims, head_w, d_model 96) gate against THIS anchor next.
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a13\track2_a13.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py
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
set RWKV_STEP_TRACE=scratchpad/track2_a13/track2_a13_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a13/t2a13_grad_stats_ws.json

echo ===== track2_a13 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a13\track2_a13_ws_trace.jsonl scratchpad\track2_a13\track2_a13_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, A9 recipe + ZERO_FEATURES=22 re-anchor, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a13/track2_a13_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a13/t2a13_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a13 t2a13ws t2a13d scratchpad/track2_a13/track2_a13_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a13/track2_a13_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a13.jsonl result\RWKV-P-track2_a13.jsonl result\RWKV-track2_a13-s0.jsonl result\RWKV-P-track2_a13-s0.jsonl result\RWKV-track2_a13.nanskip.jsonl result\RWKV-track2_a13-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a13 t2a13d scratchpad/track2_a13/track2_a13_eval.toml RWKV-track2_a13 RWKV-P-track2_a13 5001 7500 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a13/track2_a13_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === RE-ANCHOR COMPARISON vs A9 (INFORMATIONAL, not a gate) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a13.jsonl --cand-imm result/RWKV-P-track2_a13.jsonl --champ-ahead result/RWKV-track2_a9.jsonl --champ-imm result/RWKV-P-track2_a9.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%; re-anchor -- informational) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
