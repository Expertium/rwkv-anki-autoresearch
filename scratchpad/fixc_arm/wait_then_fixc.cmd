@echo off
REM ===========================================================================================
REM After the e2s re-base: finish the fixc control db, then run the fixc arm.
REM
REM WHY AFTER AND NOT ALONGSIDE. The rebuild is CPU-only, so it looks like free parallelism. It
REM is not: the training step is about 85% CPU-dispatch-bound, and a measurement today showed a
REM 6-thread rebuild running beside the teacher dump cost 1.64x (0.632 steps/s alone versus 0.385
REM with it). Serialising is faster in total, and the fixc arm cannot start before the re-base
REM finishes anyway.
REM
REM The rebuild is RESUMABLE -- data_processing skips users already present -- so the copy that
REM was paused earlier today simply continues where it stopped.
REM
REM Anchored findstr; the unanchored form matches this file's own prose and fires instantly.
REM No angle brackets or arrows in REM lines.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fixc_arm\wait_fixc.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\e2s_rebase\e2s_rebase.log

echo ===== fixc chain armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo e2s re-base finished %DATE% %TIME% >> "%LOG%"

REM The re-base must have SUCCEEDED. A failed re-base means the interval pair has no treatment
REM arm, and running the control alone produces a number with nothing to compare it to.
findstr /B /C:"DONE_EXIT_0" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  echo REBASE_FAILED -- not running the control without its treatment >> "%LOG%"
  echo DONE_EXIT_53 %DATE% %TIME% >> "%LOG%"
  exit /b 53
)

echo --- resuming the fixc control rebuild %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\run_fixc_rebuild.cmd
findstr /B /C:"DONE_EXIT_0" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\fixc_rebuild.log >nul 2>&1
if errorlevel 1 (
  echo FIXC_REBUILD_FAILED >> "%LOG%"
  echo DONE_EXIT_54 %DATE% %TIME% >> "%LOG%"
  exit /b 54
)
echo fixc dbs built %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fixc_arm\run_fixc_arm.cmd
echo fixc arm returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
