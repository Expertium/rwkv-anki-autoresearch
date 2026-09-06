@echo off
REM Waits for muonscale (scratchpad/muonscale/muonscale.log) to write its anchored terminal marker (ANY DONE_EXIT_:
REM eqw needs a free GPU, not a muonscale success), lets auto_control.py apply muonscale's both-modes gate
REM and regenerate eqw's runner on the right base, then runs it. Refuses to launch if auto_control.py fails.
REM Polls every 2 min. Writes wait_eqw.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eqw\wait_eqw.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\muonscale\muonscale.log
echo ===== eqw waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitms
if not exist "%G0%" (
  ping -n 121 127.0.0.1 >nul
  goto waitms
)
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitms
)
echo muonscale reported, choosing the base %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/eqw/auto_control.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo auto_control FAILED -- not launching eqw %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_65 %DATE% %TIME% >> "%LOG%"
  exit /b 65
)
echo launching eqw %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eqw\run_eqw.cmd
echo run_eqw returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
