@echo off
REM ============================================================================
REM TRACK-2 A9 BUNDLE: note 2L->1L (new arch module, -82,957; HALVES per-note
REM deploy state) + L0 mixer strips user_id:0 + preset_id:0 (A8 grad report's #1
REM and #6 lowest-saliency units). 1,617,975 -> 1,468,724 params (-149,251 =
REM -9.22% vs A8, -46.8% vs the original 2.76M). STRIP_CMIX = 9 entries (note_id:1
REM left with the removed layer; note.L0's mixer KEPT -- stability caution).
REM Gate vs A8 (0.300380/0.269006 full-range): ratio 100k*dLL/dparams <= 0.0001
REM BOTH modes on the VAL half 5001-7500 (allowed 0.000149/mode), pairing via
REM --intersect. VPRUNE_MIN_STEP=6000. Watch A8's stability item (val NaNs/RESETs).
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a9\track2_a9.log
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
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_a9/track2_a9_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a9/t2a9_grad_stats_ws.json

echo ===== track2_a9 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a9\track2_a9_ws_trace.jsonl scratchpad\track2_a9\track2_a9_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, d=128 A8 recipe + note1L + 2 L0 cmix strips, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a9/track2_a9_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a9/t2a9_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a9 t2a9ws t2a9d scratchpad/track2_a9/track2_a9_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a9/track2_a9_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a9.jsonl result\RWKV-P-track2_a9.jsonl result\RWKV-track2_a9-s0.jsonl result\RWKV-P-track2_a9-s0.jsonl result\RWKV-track2_a9.nanskip.jsonl result\RWKV-track2_a9-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a9 t2a9d scratchpad/track2_a9/track2_a9_eval.toml RWKV-track2_a9 RWKV-P-track2_a9 5001 7500 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a9/track2_a9_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs A8 champion (val half; ratio gate judged at record time) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a9.jsonl --cand-imm result/RWKV-P-track2_a9.jsonl --champ-ahead result/RWKV-track2_a8.jsonl --champ-imm result/RWKV-P-track2_a8.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
