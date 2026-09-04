@echo off
REM Waits for durdrop (scratchpad/durdrop/durdrop.log) to write its anchored terminal marker (ANY
REM DONE_EXIT_: ordcut needs a free GPU, not a durdrop success), then lets auto_control.py apply the
REM mechanical gate (durdrop ACCEPT => ordcut is regenerated on durdrop's recipe), then runs ordcut.
REM Refuses to launch if auto_control.py fails (the chosen runner must preflight). Polls every 2 min.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ordcut\wait_ordcut.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\durdrop\durdrop.log
echo ===== ordcut waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitdd
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitdd
)
echo durdrop reported, choosing the control %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/ordcut/auto_control.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo auto_control FAILED -- not launching ordcut %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_65 %DATE% %TIME% >> "%LOG%"
  exit /b 65
)
echo launching ordcut %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ordcut\run_ordcut.cmd
echo run_ordcut returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
