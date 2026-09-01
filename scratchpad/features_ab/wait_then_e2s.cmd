@echo off
REM ===========================================================================================
REM Park the E2S INTERVAL ARM behind the GPU queue, and gate it on the integrity check.
REM
REM QUEUE: featA2 eval (running, the END-TO-END control) then the MAX sweep (phase 1) then this.
REM This waits on the SWEEP's terminal marker, which is itself parked behind the eval, so one
REM anchored wait is enough. Anchored `findstr /B /C:"DONE_EXIT_"` -- the unanchored form matches
REM the waiter's own prose and fires instantly.
REM
REM ---- PHASE 0 IS THE POINT, AND IT IS NOT BOILERPLATE ----
REM
REM Andrew 2026-08-30 found that the srs-benchmark interval comparison was confounded by its
REM DENOMINATOR: `delta_t > 0` deleted 0.172% of reviews whose corrected gap floored to zero, and
REM those rows were 2.7x easier than average (6.09% failure versus 16.14%). Deleting the easiest
REM rows raises mean logloss by itself. Two thirds of the reported effect was that artifact.
REM
REM Our pipeline should not share it: no `delta_t > 0` filter, every row kept and only marked via
REM label_is_equalize, and the e2s tomls reuse the same label_filter_db and user ranges. The TEST
REM db already confirms it -- 170,384 entries in both arms. But "should not" is a prediction, and
REM gate #1 exists because a review-count change is the signature of a pipeline bug. So the TRAIN
REM db gets the same check, and this runner REFUSES to spend 9 hours if it fails.
REM
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\wait_e2s.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch\sweep_max3.log
set REBUILDLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\e2s_rebuild.log

echo ===== e2s waiter armed %DATE% %TIME% ===== >> "%LOG%"

REM ---- wait for the rebuild that BUILDS this arm's dbs ----
:waitrebuild
findstr /B /C:"DONE_EXIT_" "%REBUILDLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitrebuild
)
echo rebuild finished %DATE% %TIME% >> "%LOG%"

REM ---- the rebuild must have SUCCEEDED, not merely ended ----
findstr /B /C:"DONE_EXIT_0" "%REBUILDLOG%" >nul 2>&1
if errorlevel 1 (
  echo REBUILD_FAILED -- refusing to train on a partial db >> "%LOG%"
  echo DONE_EXIT_51 %DATE% %TIME% >> "%LOG%"
  exit /b 51
)

REM ---- wait for the GPU queue ahead of this arm ----
:waitsweep
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitsweep
)
echo sweep finished, GPU free %DATE% %TIME% >> "%LOG%"

REM ---- PHASE 0: gate #1. Identical review counts, or this experiment is not interpretable. ----
.venv\Scripts\python.exe scratchpad/features_rebuild/compare_db.py train_db_5k_h1_fix F:/rwkv_lmdb/train_db_5k_h1_e2s >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo SIZE_GATE_FAILED -- the arms do not score the same reviews >> "%LOG%"
  echo DONE_EXIT_52 %DATE% %TIME% >> "%LOG%"
  exit /b 52
)
echo SIZE_GATE_PASS %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\e2s\run_e2s.cmd
echo e2s runner returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
