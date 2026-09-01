@echo off
REM ===========================================================================================
REM Park the E2S CHAMPION RE-BASE behind the MAX sweep.
REM
REM Anchored findstr -- the unanchored form matches the waiter's own prose and fires instantly.
REM No angle brackets or arrows in REM lines.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\e2s_rebase\wait_rebase.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch\sweep_max3.log

echo ===== e2s re-base waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 61 127.0.0.1 >nul
  goto waitprev
)
echo sweep finished, GPU free %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\e2s_rebase\run_e2s_rebase.cmd
echo re-base returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
