@echo off
REM Budget-curve chain, behind the QAT-KD arm: dry w10b2, dry w10b5, then w10b2, then w10b5.
REM Re-pointed 2026-09-10 when Andrew approved QAT-KD: it was armed behind cmixgraft and was
REM stopped while still looping in gate 0, so the KD arm could take the slot before it.
REM
REM GATE 0 polls the QAT-KD WAITER's own log, wait_qatkd.log -- NOT w10qatkd.log, which a refused
REM QAT-KD dry run never creates, so a waiter on it would idle the GPU forever. Its codes:
REM   DONE_EXIT_110        arm 2's eval FAILED upstream: refuse -- it must be resumed first
REM   DONE_EXIT_0, 111-113 the KD arm ran, or its dry run refused: the GPU is free, proceed
REM GATE 1 runs BOTH dry runs before either real run (fail fast): each must end DONE_EXIT_0 with no
REM swallowed exception in its decay log and both result files. RWKV_DECAY_FROM_STEP is a code
REM path no finished run has exercised, which is exactly the dry-run trigger.
REM Then w10b2 and w10b5, each ~6.5 h (2-epoch decay + a 300-user eval).
REM
REM Anything built before this fires can displace it. STOP THE WAITER'S cmd FIRST, then its
REM children: stopping its ping or console host first lets the dying cmd run one more step with
REM broken exit codes (2026-09-10: it took a refusal branch; the launch branch was as likely).
REM This log carries the waiter's own DONE_EXIT_ lines (the repo convention); nothing polls it.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10b2\wait_budget.log
set PREV=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qatkd\wait_qatkd.log
set R=C:\Users\Andrew\rwkv-anki-autoresearch
echo ===== budget-curve waiter armed on the QAT-KD waiter %DATE% %TIME% ===== >> "%WLOG%"

:waitprev
if not exist "%PREV%" goto sleepprev
findstr /B /C:"DONE_EXIT_" "%PREV%" >nul 2>&1
if errorlevel 1 goto sleepprev
goto gotprev
:sleepprev
ping -n 61 127.0.0.1 >nul
goto waitprev

:gotprev
findstr /B /C:"DONE_EXIT_110 " "%PREV%" >nul 2>&1
if not errorlevel 1 (
  echo arm 2 eval FAILED per the QAT-KD waiter -- not launching the budget curve %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_100 %DATE% %TIME% >> "%WLOG%"
  exit /b 100
)
echo QAT-KD chain done -- 120 s for its workers, then the two dry runs %DATE% %TIME% >> "%WLOG%"
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
