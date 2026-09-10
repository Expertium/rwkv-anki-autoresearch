@echo off
REM Budget-curve chain, behind cmixgraft: dry w10b2, dry w10b5, then w10b2, then w10b5.
REM
REM GATE 0 polls the cmixgraft WAITER's own log, wait_cmixgraft.log -- NOT cmixgraft.log. If the
REM cmixgraft dry run refuses, cmixgraft.log is never created, and a waiter on it would idle the
REM GPU forever. Reading the waiter's code instead:
REM   DONE_EXIT_90       arm 2's eval FAILED: refuse -- it must be resumed first, and the curve
REM                      would book the GPU for ~13 h
REM   DONE_EXIT_0, 92-95 cmixgraft ran, or its dry run refused: the GPU is free, proceed
REM GATE 1 runs BOTH dry runs before either real run (fail fast): each must end DONE_EXIT_0 with no
REM swallowed exception in its decay log and both result files. RWKV_DECAY_FROM_STEP is a code
REM path no finished run has exercised, which is exactly the dry-run trigger.
REM Then w10b2 and w10b5, each ~6.5 h (2-epoch decay + a 300-user eval).
REM
REM A QAT-tax retry built before this fires can displace it: stop THIS waiter while it is still
REM looping in gate 0 (a called .cmd is not open until the call).
REM This log carries the waiter's own DONE_EXIT_ lines (the repo convention); nothing polls it.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10b2\wait_budget.log
set PREV=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\cmixgraft\wait_cmixgraft.log
set R=C:\Users\Andrew\rwkv-anki-autoresearch
echo ===== budget-curve waiter armed on the cmixgraft waiter %DATE% %TIME% ===== >> "%WLOG%"

:waitprev
if not exist "%PREV%" goto sleepprev
findstr /B /C:"DONE_EXIT_" "%PREV%" >nul 2>&1
if errorlevel 1 goto sleepprev
goto gotprev
:sleepprev
ping -n 61 127.0.0.1 >nul
goto waitprev

:gotprev
findstr /B /C:"DONE_EXIT_90 " "%PREV%" >nul 2>&1
if not errorlevel 1 (
  echo arm 2 eval FAILED per the cmixgraft waiter -- not launching the budget curve %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_100 %DATE% %TIME% >> "%WLOG%"
  exit /b 100
)
echo cmixgraft chain done -- 120 s for its workers, then the two dry runs %DATE% %TIME% >> "%WLOG%"
ping -n 121 127.0.0.1 >nul

call %R%\scratchpad\w10b2dry\run_w10b2dry.cmd
findstr /B /C:"DONE_EXIT_0 " "%R%\scratchpad\w10b2dry\w10b2dry.log" >nul 2>&1
if errorlevel 1 (
  echo w10b2 dry run ended WITHOUT success -- not launching the curve %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_101 %DATE% %TIME% >> "%WLOG%"
  exit /b 101
)
findstr /C:"Traceback" /C:"Exception caught" "%R%\scratchpad\w10b2dry\decay_*.log" >nul 2>&1
if not errorlevel 1 (
  echo w10b2 dry decay swallowed an exception -- a HOLLOW run, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_102 %DATE% %TIME% >> "%WLOG%"
  exit /b 102
)
if not exist "%R%\result\RWKV-P-w10b2dry.jsonl" (
  echo w10b2 dry eval produced no result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_103 %DATE% %TIME% >> "%WLOG%"
  exit /b 103
)

call %R%\scratchpad\w10b5dry\run_w10b5dry.cmd
findstr /B /C:"DONE_EXIT_0 " "%R%\scratchpad\w10b5dry\w10b5dry.log" >nul 2>&1
if errorlevel 1 (
  echo w10b5 dry run ended WITHOUT success -- not launching the curve %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_104 %DATE% %TIME% >> "%WLOG%"
  exit /b 104
)
findstr /C:"Traceback" /C:"Exception caught" "%R%\scratchpad\w10b5dry\decay_*.log" >nul 2>&1
if not errorlevel 1 (
  echo w10b5 dry decay swallowed an exception -- a HOLLOW run, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_105 %DATE% %TIME% >> "%WLOG%"
  exit /b 105
)
if not exist "%R%\result\RWKV-P-w10b5dry.jsonl" (
  echo w10b5 dry eval produced no result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_106 %DATE% %TIME% >> "%WLOG%"
  exit /b 106
)

echo both dry runs clean -- launching w10b2 %DATE% %TIME% >> "%WLOG%"
call %R%\scratchpad\w10b2\run_w10b2.cmd
echo run_w10b2.cmd returned %ERRORLEVEL% -- launching w10b5 %DATE% %TIME% >> "%WLOG%"
call %R%\scratchpad\w10b5\run_w10b5.cmd
echo run_w10b5.cmd returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
