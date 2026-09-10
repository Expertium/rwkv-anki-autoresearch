@echo off
REM QAT-KD chain, behind cmixgraft: the w10qatkd DRY RUN, then w10qatkd (arm 2 + distillation from
REM the full-precision twin, Andrew 2026-09-10: "Yep, do KD").
REM
REM GATE 0 polls the cmixgraft WAITER's own log, wait_cmixgraft.log -- NOT cmixgraft.log, which a
REM refused cmixgraft dry run never creates. Reading the waiter's code:
REM   DONE_EXIT_90       arm 2's eval FAILED: refuse (110) -- it must be resumed first
REM   DONE_EXIT_0, 92-95 cmixgraft ran, or its dry run refused: the GPU is free, proceed
REM GATE 1 -- the dry run (0.05 epochs, 20 users) must end DONE_EXIT_0 with no swallowed exception
REM in its decay log and a result file. Its runner already refuses unless the decay log names the
REM teacher (81) and the probe loads the LEARNED catalogs (72, 73). New phase shape: QAT plus a
REM live teacher, which no finished run has combined.
REM Downstream, wait_qatkd_then_budget.cmd polls THIS log and refuses only on 110.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qatkd\wait_qatkd.log
set PREV=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\cmixgraft\wait_cmixgraft.log
set R=C:\Users\Andrew\rwkv-anki-autoresearch
echo ===== QAT-KD waiter armed on the cmixgraft waiter %DATE% %TIME% ===== >> "%WLOG%"

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
  echo arm 2 eval FAILED per the cmixgraft waiter -- not launching the QAT-KD arm %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_110 %DATE% %TIME% >> "%WLOG%"
  exit /b 110
)
echo cmixgraft chain done -- 120 s for its workers, then the QAT-KD dry run %DATE% %TIME% >> "%WLOG%"
ping -n 121 127.0.0.1 >nul

call %R%\scratchpad\w10qatkddry\run_w10qatkddry.cmd
findstr /B /C:"DONE_EXIT_0 " "%R%\scratchpad\w10qatkddry\w10qatkddry.log" >nul 2>&1
if errorlevel 1 (
  echo QAT-KD dry run ended WITHOUT success -- not launching w10qatkd %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_111 %DATE% %TIME% >> "%WLOG%"
  exit /b 111
)
findstr /C:"Traceback" /C:"Exception caught" "%R%\scratchpad\w10qatkddry\decay_*.log" >nul 2>&1
if not errorlevel 1 (
  echo QAT-KD dry decay swallowed an exception -- a HOLLOW run, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_112 %DATE% %TIME% >> "%WLOG%"
  exit /b 112
)
if not exist "%R%\result\RWKV-P-w10qatkddry.jsonl" (
  echo QAT-KD dry eval produced no result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_113 %DATE% %TIME% >> "%WLOG%"
  exit /b 113
)

echo dry run clean -- launching w10qatkd %DATE% %TIME% >> "%WLOG%"
call %R%\scratchpad\w10qatkd\run_w10qatkd.cmd
echo run_w10qatkd.cmd returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
