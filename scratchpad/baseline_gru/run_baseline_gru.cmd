@echo off
REM ============================================================================
REM BASELINE GRU (Andrew 2026-07-23): classic GRU streams (rwkv/model/
REM rnn_baseline.py, RWKV_BASELINE_CELL=gru, hidden=128) replacing the RWKV-7
REM stacks -- same 5-stream hierarchy/depths (card2/deck4/note1/preset3/user3),
REM same trunk/heads/pipeline/budget/seed. 1,556,496 params (~A13's 1,468,724).
REM PURPOSE: is RWKV-7's complexity needed? Tail comparison vs A13 INFORMATIONAL.
REM Masking semantics smoke-verified vs a stepwise reference (interior skips).
REM RWKV_NO_JIT=1 MANDATORY; CUBLAS_WORKSPACE_CONFIG for cuDNN RNN determinism.
REM vprune OFF (cross-architecture val comparison would false-kill). NaN guard on.
REM STEP 0 waits for A14's DONE_EXIT; STEP 0.5 = 40-step E2E sanity.
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\baseline_gru\baseline_gru.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_NO_JIT=1
set RWKV_BASELINE_CELL=gru
set RWKV_ARCH_MODULE=scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_ZERO_FEATURES=22
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/baseline_gru/baseline_gru_ws_trace.jsonl

echo ===== baseline_gru START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for track-2 A14 to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a14\track2_a14.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo A14 done -- starting baseline GRU %TIME% >> "%LOG%"

echo === STEP 0.5: 40-step E2E sanity (baseline wiring + cuDNN determinism) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/baseline_gru/baseline_gru_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)
set RWKV_MAX_STEPS=

del /Q scratchpad\baseline_gru\baseline_gru_ws_trace.jsonl scratchpad\baseline_gru\baseline_gru_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, GRU streams, vprune OFF) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/baseline_gru/baseline_gru_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/baseline_gru bgruws bgrud scratchpad/baseline_gru/baseline_gru_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/baseline_gru/baseline_gru_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-baseline_gru.jsonl result\RWKV-P-baseline_gru.jsonl result\RWKV-baseline_gru-s0.jsonl result\RWKV-P-baseline_gru-s0.jsonl result\RWKV-baseline_gru.nanskip.jsonl result\RWKV-baseline_gru-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/baseline_gru bgrud scratchpad/baseline_gru/baseline_gru_eval.toml RWKV-baseline_gru RWKV-P-baseline_gru 5001 7500 >> "%LOG%" 2>&1
echo === EVAL (single process, GRU streams) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/baseline_gru/baseline_gru_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === COMPARISON vs A13 champion (INFORMATIONAL -- RWKV vs GRU at ~equal params) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-baseline_gru.jsonl --cand-imm result/RWKV-P-baseline_gru.jsonl --champ-ahead result/RWKV-track2_a13.jsonl --champ-imm result/RWKV-P-track2_a13.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%; baseline -- informational) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
