@echo off
REM =========================================================================================
REM Arm featB behind featA. featA writes its terminal marker BEFORE endlocal, so the marker
REM really does appear in its log (the omission that stranded a 4-job chain on 2026-08-18).
REM
REM ANCHORED grep: findstr /B /C:"DONE_EXIT_". The unanchored form matches any line that merely
REM MENTIONS the token, including a waiter's own startup echo, and fires instantly. That is why
REM the echo below says TERMINAL MARKER and never spells the token.
REM
REM featB needs the GENERATION-2 dbs, so this must not be armed until run_rebuild2.cmd has
REM written its own success marker and check_db has passed at width 46.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featA\featA.log
set NEXT=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featB\run_featB.cmd
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\wait_featB.log

echo waiter armed %DATE% %TIME% -- polling featA for its TERMINAL MARKER > "%WLOG%"

:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 60 /nobreak >nul
goto loop

:go
echo featA finished %DATE% %TIME% >> "%WLOG%"
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >> "%WLOG%"
if not exist "%NEXT%" (
  echo featB runner MISSING -- not launching %DATE% %TIME% >> "%WLOG%"
  exit /b 9
)
echo launching featB %DATE% %TIME% >> "%WLOG%"
call "%NEXT%"
echo featB returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
REM The waiter's own terminal marker, in its OWN log -- featB writes its separate one in
REM featB.log. Before endlocal, or %WLOG% expands to empty and the marker goes to "".
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
