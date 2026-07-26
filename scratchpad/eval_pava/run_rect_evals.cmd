@echo off
REM ============================================================================
REM Rectified evals (RWKV_EVAL_PAVA=1) for the DEPLOY metric, 2026-07-26.
REM
REM iter 31's own eval leg is UNRECTIFIED -- its .cmd was written before the flag
REM existed, and a running batch file must not be edited (cmd.exe re-reads it at a
REM saved byte offset). That unrectified number is the primary gate, since it is
REM directly comparable to A18's existing jsonls. THIS script produces the other
REM metric: both models scored as they would actually ship.
REM
REM A18 has no pava_theta, so RWKV_PAVA_LAMBDA stays unset and the rectifier falls
REM back to classic p=1 PAVA -- the honest comparison, since that IS what shipping
REM A18 would do. iter 31 uses its learned powers.
REM
REM Separate _rect output tags: the unrectified jsonls are the primary gate and
REM must not be clobbered.
REM
REM Parks on iter 31's DONE_EXIT (no co-tenant GPU work during a gate-critical run).
REM Launch:  .\scratchpad\detach.ps1 -Script scratchpad\eval_pava\run_rect_evals.cmd
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eval_pava
set LOG=%DIR%\rect_evals.log
set STAMP=%RANDOM%%RANDOM%
set WAITLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter31_algo\iter31_algo.log

echo ===== rect_evals START %DATE% %TIME% ===== > "%LOG%"

echo === WAIT for iter 31 DONE_EXIT %TIME% === >> "%LOG%"
:waitloop
findstr /C:"DONE_EXIT" "%WAITLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto ready
timeout /t 120 /nobreak >nul
goto waitloop
:ready
echo iter 31 finished, GPU free %TIME% >> "%LOG%"

REM ---------------------------------------------------------------- A18 (p=1)
setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_EVAL_PAVA=1
set RWKV_PROBE_DUR=0.0

del /Q result\RWKV-track2_a18_rect.jsonl result\RWKV-P-track2_a18_rect.jsonl result\RWKV-track2_a18_rect.nanskip.jsonl 2>nul
echo === A18 rectified eval toml %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a18 t2a18d scratchpad/eval_pava/a18_rect_eval.toml RWKV-track2_a18_rect RWKV-P-track2_a18_rect 5001 7500 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_A18TOML %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
echo === A18 rectified eval %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/eval_pava/a18_rect_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\a18_rect_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_A18EVAL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
echo A18 RECT OK %TIME% >> "%LOG%"
endlocal

REM ------------------------------------------------------- iter 31 (learned p)
if not exist scratchpad\iter31_algo\iter31d_5586.pth (
  echo SKIP iter31 rect eval: no decay ckpt ^(run did not finish^) %TIME% >> "%LOG%"
  echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
  exit /b 0
)
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
REM PAVA_LAMBDA must be set: it is what creates pava_theta, and load_state_dict is
REM strict -- without it the iter-31 checkpoint will not open at all.
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_MUON=1
set RWKV_EVAL_PAVA=1
set RWKV_PROBE_DUR=0.0

del /Q result\RWKV-iter31_algo_rect.jsonl result\RWKV-P-iter31_algo_rect.jsonl result\RWKV-iter31_algo_rect.nanskip.jsonl 2>nul
echo === iter31 rectified eval toml %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter31_algo iter31d scratchpad/eval_pava/iter31_rect_eval.toml RWKV-iter31_algo_rect RWKV-P-iter31_algo_rect 5001 7500 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_I31TOML %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === iter31 rectified eval %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/eval_pava/iter31_rect_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\iter31_rect_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_I31EVAL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo ITER31 RECT OK %TIME% >> "%LOG%"
endlocal

echo === SANITY: imm must be BIT-IDENTICAL rect vs unrect (probes are skip rows) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/eval_pava/check_imm_identical.py result/RWKV-P-iter31_algo.jsonl result/RWKV-P-iter31_algo_rect.jsonl >> "%LOG%" 2>&1

echo === GATE on the RECTIFIED pair (val half, paired) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter31_algo_rect.jsonl --cand-imm result/RWKV-P-iter31_algo_rect.jsonl --champ-ahead result/RWKV-track2_a18_rect.jsonl --champ-imm result/RWKV-P-track2_a18_rect.jsonl >> "%LOG%" 2>&1

echo === For reference, the UNRECTIFIED gate (primary) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter31_algo.jsonl --cand-imm result/RWKV-P-iter31_algo.jsonl --champ-ahead result/RWKV-track2_a18.jsonl --champ-imm result/RWKV-P-track2_a18.jsonl >> "%LOG%" 2>&1

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
