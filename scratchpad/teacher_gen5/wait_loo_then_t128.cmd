@echo off
REM Waits for the LOO sweep (scratchpad/feat_loo/loo.log) to write its anchored terminal marker --
REM ANY DONE_EXIT_ is fine here, because the timing run only needs a free GPU, not a LOO success --
REM then runs the d=128 teacher timing (about 15 minutes). Polls every 2 minutes.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\teacher_gen5\wait_t128.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_loo\loo.log
echo ===== t128 timing waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitloo
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitloo
)
echo LOO reported, launching timing %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\teacher_gen5\run_t128_timing.cmd
echo timing returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
