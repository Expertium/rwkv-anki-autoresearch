@echo off
REM ============================================================================
REM Probe-insertion NOISE control (RWKV_EVAL_PAVA=3), 2026-07-26.
REM
REM WHY THIS EXISTS. Probes are skip rows and the token shift steps over them, so in EXACT
REM arithmetic inserting them changes nothing. In bf16 it does: +4 rows per scored review
REM inflates the batch ~30%, re-buckets sequences by length, and reorders bf16 reductions.
REM Measured on A18 (n=2500) via `imm`, the channel the rectifier cannot reach: mean +0.000280,
REM scaling with recurrence length (1.98e-4 at ~4.7k reviews/user -> 3.97e-4 at ~179k) and
REM one-signed (62% -> 78% of users worse) because LogLoss is CONVEX -- zero-mean noise on a
REM prediction raises it.
REM
REM So `mode2 - mode0` would charge the duration change for that noise too. Mode 3 inserts the
REM IDENTICAL probes and substitutes nothing, giving the clean split:
REM     mode3 - mode0 = probe-insertion noise
REM     mode2 - mode3 = cost of zeroing the current-row duration   <- what Andrew asked for
REM     mode1 - mode2 = cost of the PAVA pooling itself
REM
REM PARKED BEHIND ITER 32, NOT BEFORE IT. iter32's .cmd is already parked on mode2_diag's
REM DONE_EXIT_ and a running batch file must never be edited, so parking this on mode2 as well
REM would run it CO-TENANT with iter 32's training. Waiting for iter 32 costs ~8 h of latency on
REM a diagnostic nobody is blocked on, and costs nothing else.
REM
REM Env is byte-identical to run_mode2_diag.cmd except RWKV_EVAL_PAVA, and the user range matches
REM (5001-5500) so all four modes pair on the same users.
REM
REM Launch:  .\scratchpad\detach.ps1 -Script C:\...\scratchpad\eval_pava\run_mode3_noise.cmd
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eval_pava
set LOG=%DIR%\mode3_noise.log
set STAMP=%RANDOM%%RANDOM%
set WAITLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter32_kd\iter32_kd.log

echo ===== mode3_noise START %DATE% %TIME% ===== > "%LOG%"

REM /B anchored, and this message deliberately avoids the token itself.
echo === WAIT for iter 32 to finish %TIME% === >> "%LOG%"
:waitloop
findstr /B /C:"DONE_EXIT_" "%WAITLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto ready
timeout /t 300 /nobreak >nul
goto waitloop
:ready
echo iter 32 finished, GPU free %TIME% >> "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_GRU_HEAD=3
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_MUON=1
set RWKV_EVAL_PAVA=3
set RWKV_PROBE_DUR=0.0

del /Q result\RWKV-iter31_algo_noise.jsonl result\RWKV-P-iter31_algo_noise.jsonl result\RWKV-iter31_algo_noise.nanskip.jsonl 2>nul
echo === mode3 eval toml %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter31_algo iter31d scratchpad/eval_pava/iter31_noise_eval.toml RWKV-iter31_algo_noise RWKV-P-iter31_algo_noise 5001 5500 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOML %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
echo === mode3 eval (users 5001-5500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/eval_pava/iter31_noise_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\iter31_noise_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVAL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
if not exist result\RWKV-iter31_algo_noise.jsonl (
  echo DONE_EXIT_NOOUTPUT %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo MODE3 EVAL OK %TIME% >> "%LOG%"
endlocal

if not exist result\RWKV-iter31_algo_raw.jsonl (
  echo SKIP decomposition: no mode-2 jsonl %TIME% >> "%LOG%"
  echo DONE_EXIT_NOMODE2 %DATE% %TIME% >> "%LOG%"
  exit /b 0
)
echo === FULL 4-MODE DECOMPOSITION %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/eval_pava/decompose_duration.py ^
  --mode0 result/RWKV-iter31_algo.jsonl ^
  --mode1 result/RWKV-iter31_algo_rect.jsonl ^
  --mode2 result/RWKV-iter31_algo_raw.jsonl ^
  --mode3 result/RWKV-iter31_algo_noise.jsonl ^
  --imm0 result/RWKV-P-iter31_algo.jsonl ^
  --imm1 result/RWKV-P-iter31_algo_rect.jsonl ^
  --imm2 result/RWKV-P-iter31_algo_raw.jsonl >> "%LOG%" 2>&1

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
