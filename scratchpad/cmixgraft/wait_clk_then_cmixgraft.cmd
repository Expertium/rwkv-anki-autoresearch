@echo off
REM cmixgraft chain v2: behind the LEAK-FREE BRANCH waiter, then the cmixgraft DRY RUN, then cmixgraft.
REM Re-pointed 2026-09-10 22:30 (v1 = wait_qateval_then_cmixgraft.cmd, stopped while it looped in
REM gate 0) so w10clk can take the slot after arm 2's eval when phase L says MATERIAL.
REM
REM GATE 0 -- polls the leak-free waiter's own log, wait_clk.log. Its codes:
REM   DONE_EXIT_130                           arm 2's eval FAILED upstream: REFUSE with 90, as v1
REM   DONE_EXIT_0 (ran), 137 (skipped), 131-136  the GPU is free: proceed
REM A MISSING log means WAIT.
REM
REM GATE 1 -- the dry run (0.05 epochs, 20 users) must end DONE_EXIT_0 with no Traceback and no
REM swallowed batch exception in its decay log, and both result files. It is owed because this
REM is a NEW PHASE SHAPE: the first decay from a GRAFTED checkpoint whose architecture differs
REM from ws10's, loading a NAME-MAPPED optimizer state (expand_optim.py). The dry runner
REM truncates its log at START, so an earlier attempt's marker cannot survive into this check.
REM
REM This log carries the waiter's own DONE_EXIT_ lines (the repo convention); nothing polls it.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\cmixgraft\wait_cmixgraft.log
set PREV=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10clk\wait_clk.log
set DRYLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\cmixgraftdry\cmixgraftdry.log
set DRYDIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\cmixgraftdry
echo ===== cmixgraft waiter v2 armed on the leak-free branch waiter %DATE% %TIME% ===== >> "%WLOG%"

:waitprev
if not exist "%PREV%" goto sleepprev
findstr /B /C:"DONE_EXIT_" "%PREV%" >nul 2>&1
if errorlevel 1 goto sleepprev
goto gotprev
:sleepprev
ping -n 61 127.0.0.1 >nul
goto waitprev

:gotprev
findstr /B /C:"DONE_EXIT_130 " "%PREV%" >nul 2>&1
if not errorlevel 1 (
  echo arm 2 eval FAILED per the leak-free waiter -- not launching cmixgraft %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_90 %DATE% %TIME% >> "%WLOG%"
  exit /b 90
)
echo leak-free waiter done -- 120 s for its workers, then the cmixgraft dry run %DATE% %TIME% >> "%WLOG%"
ping -n 121 127.0.0.1 >nul

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\cmixgraftdry\run_cmixgraftdry.cmd
findstr /B /C:"DONE_EXIT_0 " "%DRYLOG%" >nul 2>&1
if errorlevel 1 (
  echo cmixgraft dry run ended WITHOUT success -- not launching cmixgraft %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_92 %DATE% %TIME% >> "%WLOG%"
  exit /b 92
)
findstr /C:"Traceback" /C:"Exception caught" "%DRYDIR%\decay_*.log" >nul 2>&1
if not errorlevel 1 (
  echo cmixgraft dry decay swallowed an exception -- a HOLLOW run, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_93 %DATE% %TIME% >> "%WLOG%"
  exit /b 93
)
if not exist "C:\Users\Andrew\rwkv-anki-autoresearch\result\RWKV-cmixgraftdry.jsonl" (
  echo cmixgraft dry eval produced no ahead result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_94 %DATE% %TIME% >> "%WLOG%"
  exit /b 94
)
if not exist "C:\Users\Andrew\rwkv-anki-autoresearch\result\RWKV-P-cmixgraftdry.jsonl" (
  echo cmixgraft dry eval produced no imm result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_95 %DATE% %TIME% >> "%WLOG%"
  exit /b 95
)

echo dry run clean -- launching cmixgraft %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\cmixgraft\run_cmixgraft.cmd
echo run_cmixgraft.cmd returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
