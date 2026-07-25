@echo off
REM ============================================================================
REM BASELINE LSTM v3 -- EVAL CONTINUATION (2026-07-25 07:50). The LSTM's main .cmd
REM predates last night's eval fixes, so its own EVAL phase would run WITHOUT
REM RWKV_EVAL_EMPTY_CACHE_EVERY=1 and would very likely repeat the GRU's
REM fragmentation wedge (~11 GB reserved, 0% GPU, host spill). A running .cmd
REM cannot be edited safely (cmd.exe re-reads by byte offset), so its SHELL was
REM killed after the decay python was already running; the decay finishes on its
REM own and this script takes over from there.
REM
REM STEP 0 waits for the decay's final ckpt AND for the decay python to exit.
REM Then eval (VAL 5001-7500) + informational gate vs A13, appending to the SAME
REM control log the (dead) parent used.
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
set RWKV_NO_JIT=1
set RWKV_EXIT_HARD=1
set RWKV_BASELINE_CELL=lstm
set RWKV_ARCH_MODULE=scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_ZERO_FEATURES=22
set RWKV_EVAL_EMPTY_CACHE_EVERY=1
set RWKV_RNN_PROBE_CHUNK=32768
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo === EVAL CONTINUATION armed (waiting for decay to finish) %DATE% %TIME% === >> "%LOG%"
:waitdecay
if not exist "%DIR%\blstmd_5586.pth" (
  timeout /t 60 /nobreak >nul
  goto waitdecay
)
REM ckpt exists -- give the decay process time to exit cleanly before touching the GPU
tasklist /FI "IMAGENAME eq python.exe" /FO CSV | findstr /C:"python.exe" >nul 2>&1
timeout /t 90 /nobreak >nul
echo DECAY OK (ckpt seen) %TIME% >> "%LOG%"

del /Q result\RWKV-baseline_lstm.jsonl result\RWKV-P-baseline_lstm.jsonl result\RWKV-baseline_lstm-s0.jsonl result\RWKV-P-baseline_lstm-s0.jsonl result\RWKV-baseline_lstm.nanskip.jsonl result\RWKV-baseline_lstm-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/baseline_lstm blstmd scratchpad/baseline_lstm/baseline_lstm_eval.toml RWKV-baseline_lstm RWKV-P-baseline_lstm 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
echo === EVAL (single process, LSTM streams; empty_cache every user) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/baseline_lstm/baseline_lstm_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === COMPARISON vs A13 champion (INFORMATIONAL -- RWKV vs LSTM at ~equal params) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-baseline_lstm.jsonl --cand-imm result/RWKV-P-baseline_lstm.jsonl --champ-ahead result/RWKV-track2_a13.jsonl --champ-imm result/RWKV-P-track2_a13.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%; baseline -- informational) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
