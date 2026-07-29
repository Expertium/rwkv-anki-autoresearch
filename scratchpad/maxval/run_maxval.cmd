@echo off
REM ============================================================================
REM MAX=65536 ACCURACY TEST (2026-07-29) -- the last big speed lever.
REM The sweep measured 13,899 vs 8,607 reviews/s = 1.61x. But MAX is NOT a pure
REM speed knob: groups fall 22,346 -> 10,935, i.e. HALF the optimizer steps per
REM epoch. This asks the cheap question first -- does accuracy hold at the SAME
REM LR? If yes it is free. If no, the delta quantifies how much LR/warmup
REM retuning would be needed, which is Andrew's call.
REM
REM BASELINE = iter 31 (NOT the iter-32 champion): changing MAX changes the step
REM count, which would desynchronise iter 32's step-keyed KD teacher dump. iter 31
REM is the same trunk without KD and its RECTIFIED numbers exist
REM (ahead 0.300802 / imm 0.267691).
REM The validated-neutral combo is ON, so any delta is attributable to MAX.
REM ⚠ VRAM: the sweep peaked at 9.951 GB WITHOUT compile. The 40-step sanity phase
REM is the check; a SANITYFAIL here means OOM, not a bad idea.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\maxval
set LOG=%DIR%\maxval.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.02
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_VPRUNE_REF=

REM the ADOPTED speed env (validated accuracy-neutral 2026-07-29)
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1

echo ===== MAXVAL (MAX=65536) START %DATE% %TIME% ===== > "%LOG%"

echo === SANITY 40 steps (VRAM envelope at MAX=65536 + compile) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/maxval/maxval_ws.toml > "%DIR%\sanity_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% ^(likely OOM at MAX=65536 with compile^) %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
set RWKV_MAX_STEPS=
findstr /C:"BATCHED Newton-Schulz" "%DIR%\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_MUONNOTBATCHED %DATE% %TIME% >> "%LOG%"
  exit /b 16
)
echo SANITY OK %TIME% >> "%LOG%"

set RWKV_STEP_TRACE=scratchpad/maxval/maxval_ws_trace.jsonl
del /Q scratchpad\maxval\maxval_ws_trace.jsonl scratchpad\maxval\maxval_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (MAX=65536 -> ~10,935 steps) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/maxval/maxval_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
echo WS OK %TIME% >> "%LOG%"
set RWKV_STEP_TRACE=

echo === DECAY SETUP (MAX=65536) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/maxval mvws mvd scratchpad/maxval/maxval_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 65536 > "%DIR%\decay_setup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/maxval/maxval_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo DECAY OK %TIME% >> "%LOG%"

del /Q result\RWKV-maxval.jsonl result\RWKV-P-maxval.jsonl result\RWKV-maxval-s0.jsonl result\RWKV-P-maxval-s0.jsonl 2>nul
echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/maxval mvd scratchpad/maxval/maxval_eval.toml RWKV-maxval RWKV-P-maxval 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
set RWKV_MUON_BATCHED=
set RWKV_QAT_COMPILE=
set RWKV_NO_JIT=
set RWKV_EVAL_PAVA=1
echo === EVAL (RECTIFIED) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/maxval/maxval_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === GATE: maxval vs iter 31 RECTIFIED (same trunk, no KD either side) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-maxval.jsonl --cand-imm result/RWKV-P-maxval.jsonl --champ-ahead result/RWKV-iter31_algo_rect.jsonl --champ-imm result/RWKV-P-iter31_algo_rect.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
