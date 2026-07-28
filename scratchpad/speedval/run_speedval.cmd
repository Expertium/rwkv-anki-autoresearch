@echo off
REM ============================================================================
REM SPEEDUP ACCURACY VALIDATION (2026-07-29).
REM The three measured speedups combine to 1.155x, but TWO of them perturb the
REM training trajectory and so cannot be adopted on a stopwatch alone:
REM   RWKV_MUON_BATCHED=1  -> batched bmm changes reduction order vs per-param mm
REM   RWKV_QAT_COMPILE=1   -> inductor fusion changes it too (needs RWKV_NO_JIT=1)
REM (NUM_FETCH_PROCESSES 4->2 is numerics-neutral -- DataFetcher.get(key) blocks on
REM  the specific group key -- and is baked into the toml here.)
REM
REM DESIGN: replicate the CHAMPION recipe (iter 32, full-run KD) EXACTLY, changing
REM only those flags, and gate rectified-vs-rectified against iter 32. Same MAX,
REM same data, same step count, so the teacher dump stays step-aligned.
REM PASS = both modes within noise of iter 32 (i.e. no systematic loss). A win is
REM not required; these are speed changes, not accuracy changes.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\speedval
set LOG=%DIR%\speedval.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_iter32
set WSSTEPS=22346

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

REM ---- THE CHANGES UNDER TEST ----
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1

set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.5

echo ===== SPEEDVAL START %DATE% %TIME% ===== > "%LOG%"

echo === SANITY 40 steps (KD alignment under the new flags) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/speedval/speedval_ws.toml > "%DIR%\sanity_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
set RWKV_MAX_STEPS=
findstr /C:"[kd-mix] step 1:" "%DIR%\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_KDNOTACTIVE %DATE% %TIME% >> "%LOG%"
  exit /b 14
)
findstr /C:"BATCHED Newton-Schulz" "%DIR%\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_MUONNOTBATCHED %DATE% %TIME% >> "%LOG%"
  exit /b 16
)
findstr /C:"[compile]" "%DIR%\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_COMPILENOTON %DATE% %TIME% >> "%LOG%"
  exit /b 17
)
echo SANITY OK ^(KD + batched muon + compile all confirmed active^) %TIME% >> "%LOG%"

set RWKV_STEP_TRACE=scratchpad/speedval/speedval_ws_trace.jsonl
del /Q scratchpad\speedval\speedval_ws_trace.jsonl scratchpad\speedval\speedval_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/speedval/speedval_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
if not exist scratchpad\speedval\svws_%WSSTEPS%.pth (
  echo DONE_EXIT_WSNOCKPT %DATE% %TIME% >> "%LOG%"
  exit /b 15
)
echo WS OK %TIME% >> "%LOG%"
set RWKV_STEP_TRACE=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/speedval svws svd scratchpad/speedval/speedval_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 > "%DIR%\decay_setup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/speedval/speedval_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo DECAY OK %TIME% >> "%LOG%"

del /Q result\RWKV-speedval.jsonl result\RWKV-P-speedval.jsonl result\RWKV-speedval-s0.jsonl result\RWKV-P-speedval-s0.jsonl 2>nul
echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/speedval svd scratchpad/speedval/speedval_eval.toml RWKV-speedval RWKV-P-speedval 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
REM eval WITHOUT the training-only flags; RECTIFIED to match iter 32's basis.
set RWKV_MUON_BATCHED=
set RWKV_QAT_COMPILE=
set RWKV_NO_JIT=
set RWKV_EVAL_PAVA=1
echo === EVAL (RECTIFIED) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/speedval/speedval_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === GATE: speedval vs iter 32 RECTIFIED (expect NO systematic loss) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-speedval.jsonl --cand-imm result/RWKV-P-speedval.jsonl --champ-ahead result/RWKV-iter32_kd_rect.jsonl --champ-imm result/RWKV-P-iter32_kd_rect.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
