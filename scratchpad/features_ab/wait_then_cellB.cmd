@echo off
REM ===========================================================================================
REM Park CELL B (train end-to-END, eval end-to-START) behind the MAX sweep.
REM
REM QUEUE: featA2 eval (cell A) then the MAX sweep then THIS then the e2s arm (cell C).
REM The sweep is itself parked behind the eval, so one anchored wait is enough.
REM
REM WHY CELL B RUNS BEFORE CELL C, given the total GPU time is the same either way:
REM   1. It is CHEAP -- one eval, no training -- and it DE-RISKS the queue. Cell C is a 9 h
REM      train-decay-eval chain; if it dies at hour 8 we still have a deploy-relevant number.
REM   2. It describes the model that is SHIPPED TODAY, which is the number Andrew is actively
REM      reasoning about. Learning it in ~3 h instead of ~12 h has real value.
REM
REM Anchored findstr -- the unanchored form matches the waiter's own prose and fires instantly.
REM No angle brackets or arrows in REM lines.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\wait_cellB.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch\sweep_max3.log

echo ===== cell B waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitsweep
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitsweep
)
echo sweep finished, GPU free %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\run_featA2_on_e2s.cmd
echo cell B returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
