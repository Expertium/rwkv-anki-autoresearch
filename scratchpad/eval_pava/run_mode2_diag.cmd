@echo off
REM ============================================================================
REM Duration-zeroing diagnostic (RWKV_EVAL_PAVA=2), 2026-07-26.
REM
REM Andrew: "about review duration of the most recent (current) review being zeroed
REM out - we still need to benchmark how much that affects log loss, right?"
REM
REM Modes 0 and 1 differ in TWO ways (pooling AND the zeroed current-row duration), so
REM neither can attribute its own result. Mode 2 substitutes the pressed probe WITHOUT
REM pooling, moving only the duration -> the split becomes additive.
REM
REM 500 users (5001-5500), not the full 2500: this is a DIAGNOSTIC, not a gate, and the
REM comparison is PAIRED on the identical model and users -- only one env flag moves --
REM so per-user variance is tiny compared with the cross-training comparisons the
REM 200-user subset-overfit lesson was about. ~30 min instead of ~2.4 h.
REM
REM Env is a byte-for-byte copy of the iter-31 leg of run_rect_evals.cmd except
REM RWKV_EVAL_PAVA -- anything else differing would confound the very thing being split.
REM
REM Parks on rect_evals DONE_EXIT (no co-tenant GPU work during a gate-relevant eval).
REM Launch:  .\scratchpad\detach.ps1 -Script scratchpad\eval_pava\run_mode2_diag.cmd
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eval_pava
set LOG=%DIR%\mode2_diag.log
set STAMP=%RANDOM%%RANDOM%
set WAITLOG=%DIR%\rect_evals.log

echo ===== mode2_diag START %DATE% %TIME% ===== > "%LOG%"

REM ANCHOR THE GREP WITH /B, and never write the token itself in a non-terminal line.
REM Cost of learning this: one eval launched co-tenant with a running one (2026-07-26).
REM findstr /C:"DONE_EXIT" matched rect_evals' OWN wait message ("=== WAIT for iter 31
REM DONE_EXIT ... ==="), so this loop fired instantly. Terminal lines START with the token;
REM /B requires that, which no prose line can satisfy. The message below is deliberately
REM phrased without it, so that any job parked on THIS log cannot be poisoned the same way.
echo === WAIT for rect_evals to finish %TIME% === >> "%LOG%"
:waitloop
findstr /B /C:"DONE_EXIT_" "%WAITLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto ready
timeout /t 120 /nobreak >nul
goto waitloop
:ready
echo rect_evals finished, GPU free %TIME% >> "%LOG%"

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
set RWKV_EVAL_PAVA=2
set RWKV_PROBE_DUR=0.0

del /Q result\RWKV-iter31_algo_raw.jsonl result\RWKV-P-iter31_algo_raw.jsonl result\RWKV-iter31_algo_raw.nanskip.jsonl 2>nul
echo === mode2 eval toml %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter31_algo iter31d scratchpad/eval_pava/iter31_raw_eval.toml RWKV-iter31_algo_raw RWKV-P-iter31_algo_raw 5001 5500 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOML %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
echo === mode2 eval (users 5001-5500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/eval_pava/iter31_raw_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\iter31_raw_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVAL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
if not exist result\RWKV-iter31_algo_raw.jsonl (
  echo DONE_EXIT_NOOUTPUT %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo MODE2 EVAL OK %TIME% >> "%LOG%"
endlocal

REM The waitloop fires on ANY DONE_EXIT, including rect_evals' failure paths -- correct for
REM the GPU-free question (mode 2 needs only iter 31's checkpoint), but the decomposition
REM additionally needs rect_evals' mode-1 output. Say so rather than dying in the script.
if not exist result\RWKV-iter31_algo_rect.jsonl (
  echo SKIP decomposition: no mode-1 jsonl ^(rect_evals did not finish^) %TIME% >> "%LOG%"
  echo DONE_EXIT_NOMODE1 %DATE% %TIME% >> "%LOG%"
  exit /b 0
)
echo === DECOMPOSITION %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/eval_pava/decompose_duration.py ^
  --mode0 result/RWKV-iter31_algo.jsonl ^
  --mode1 result/RWKV-iter31_algo_rect.jsonl ^
  --mode2 result/RWKV-iter31_algo_raw.jsonl ^
  --imm0 result/RWKV-P-iter31_algo.jsonl ^
  --imm1 result/RWKV-P-iter31_algo_rect.jsonl ^
  --imm2 result/RWKV-P-iter31_algo_raw.jsonl >> "%LOG%" 2>&1

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
