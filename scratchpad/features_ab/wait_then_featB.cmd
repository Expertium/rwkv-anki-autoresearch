@echo off
REM =========================================================================================
REM Arm featB behind featA, gated on BOTH preconditions.
REM
REM featB cannot start until two independent things are true, and a waiter that checks only the
REM first would launch a 7.75 h arm against a missing or half-built db:
REM   1. featA finished CLEANLY (a nonzero terminal code means there is no control arm, so the
REM      comparison has nothing to be measured against);
REM   2. the generation-2 rebuild finished cleanly AND its test db passes check_db at width 46.
REM
REM ANCHORED grep: findstr /B /C: anchors on the token at line start. The unanchored form matches
REM any line that merely MENTIONS it, including a waiter's own startup echo, and fires instantly.
REM That is why the echo below says TERMINAL MARKER and never spells the token.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featA\featA.log
set RBLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\rebuild2.log
set NEXT=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featB\run_featB.cmd
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\wait_featB.log

echo waiter armed %DATE% %TIME% -- polling featA for its TERMINAL MARKER > "%WLOG%"

:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto checks
timeout /t 60 /nobreak >nul
goto loop

:checks
echo featA finished %DATE% %TIME% >> "%WLOG%"
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >> "%WLOG%"

REM ---- precondition 1: featA exited cleanly ----
findstr /B /C:"DONE_EXIT_0 " "%PREVLOG%" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo ABORT featA did not finish cleanly -- no control arm %DATE% %TIME% >> "%WLOG%"
  exit /b 21
)

REM ---- precondition 2: the generation-2 rebuild finished cleanly ----
findstr /B /C:"DONE_EXIT_0 " "%RBLOG%" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo ABORT gen-2 rebuild did not finish cleanly %DATE% %TIME% >> "%WLOG%"
  exit /b 22
)

REM ---- precondition 3: the eval db is really there, at the right WIDTH ----
REM check_db reads entry count from metadata and the width from the first record, so this costs
REM about a second and it is the check that catches a db built without RWKV_ID_FEATURES.
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/test_db_5k_id2 2000 46 >> "%WLOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo ABORT gen-2 eval db failed check_db %DATE% %TIME% >> "%WLOG%"
  exit /b 23
)

if not exist "%NEXT%" (
  echo ABORT featB runner MISSING %DATE% %TIME% >> "%WLOG%"
  exit /b 24
)

echo launching featB %DATE% %TIME% >> "%WLOG%"
call "%NEXT%"
echo featB returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
REM The waiter's own terminal marker, in its OWN log. Before endlocal, or WLOG expands to empty.
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
