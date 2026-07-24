@echo off
REM ============================================================================
REM BASELINE LSTM v3 (Andrew 2026-07-23; v3 2026-07-24): classic LSTM streams
REM (rwkv/model/rnn_baseline.py, RWKV_BASELINE_CELL=lstm, hidden=92) replacing
REM the RWKV-7 stacks -- same 5-stream hierarchy/depths (card2/deck4/note1/
REM preset3/user3), same trunk/heads/pipeline/budget/seed. 1,488,688 params
REM (~A13's 1,468,724). v3 = pre-norm PER-LAYER RESIDUALS (see the GRU cmd
REM header for the v2 attenuation post-mortem); h 104->92 pays for the
REM per-layer projs. LSTM probes use c=0 (fresh-cell caveat, documented).
REM PURPOSE: is RWKV-7's complexity needed? Tail comparison vs A13 INFORMATIONAL.
REM Design: per-layer cuDNN + torch-RNG dropout + (layer,window) checkpoints +
REM fp32 stream weights behind boundary casts + windowed h-carry (mega users).
REM RWKV_NO_JIT=1 MANDATORY. RWKV_DETERMINISTIC=0 (cuDNN RNN backward nondet).
REM RWKV_EXIT_HARD=1 (Windows cuDNN-RNN native teardown crashes post-success).
REM LOG HYGIENE (2026-07-24): control log (%LOG%) is cmd-only; python phases
REM redirect to per-phase sublogs (straggler-worker handle inheritance blocked
REM the cmd's >> and silently killed the phase chain).
REM vprune OFF (cross-architecture val comparison would false-kill).
REM STEP 0 waits for baseline_gru's DONE_EXIT_0 (its CONTROL log, cmd-written).
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\baseline_lstm
set LOG=%DIR%\baseline_lstm.log
set STAMP=%RANDOM%%RANDOM%
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=0
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_NO_JIT=1
set RWKV_EXIT_HARD=1
set RWKV_BASELINE_CELL=lstm
set RWKV_ARCH_MODULE=scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_ZERO_FEATURES=22
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/baseline_lstm/baseline_lstm_ws_trace.jsonl

echo ===== baseline_lstm START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for baseline GRU to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT_0" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\baseline_gru\baseline_gru.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo GRU baseline done -- starting LSTM %TIME% >> "%LOG%"

echo === STEP 0.5: 40-step E2E sanity (see sanity.log) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/baseline_lstm/baseline_lstm_ws.toml > "%DIR%\sanity_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)
set RWKV_MAX_STEPS=
echo sanity OK %TIME% >> "%LOG%"

del /Q scratchpad\baseline_lstm\baseline_lstm_ws_trace.jsonl scratchpad\baseline_lstm\baseline_lstm_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, LSTM streams, vprune OFF; see ws.log) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/baseline_lstm/baseline_lstm_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=
echo WS OK %TIME% >> "%LOG%"

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/baseline_lstm blstmws blstmd scratchpad/baseline_lstm/baseline_lstm_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 > "%DIR%\decay_setup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY (see decay.log) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/baseline_lstm/baseline_lstm_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo DECAY OK %TIME% >> "%LOG%"

del /Q result\RWKV-baseline_lstm.jsonl result\RWKV-P-baseline_lstm.jsonl result\RWKV-baseline_lstm-s0.jsonl result\RWKV-P-baseline_lstm-s0.jsonl result\RWKV-baseline_lstm.nanskip.jsonl result\RWKV-baseline_lstm-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/baseline_lstm blstmd scratchpad/baseline_lstm/baseline_lstm_eval.toml RWKV-baseline_lstm RWKV-P-baseline_lstm 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
echo === EVAL (single process, LSTM streams; see eval.log) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/baseline_lstm/baseline_lstm_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === COMPARISON vs A13 champion (INFORMATIONAL -- RWKV vs LSTM at ~equal params; see gate.log) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-baseline_lstm.jsonl --cand-imm result/RWKV-P-baseline_lstm.jsonl --champ-ahead result/RWKV-track2_a13.jsonl --champ-imm result/RWKV-P-track2_a13.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%; baseline -- informational) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
