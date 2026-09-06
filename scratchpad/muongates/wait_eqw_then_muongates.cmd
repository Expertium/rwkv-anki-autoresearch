@echo off
REM Waits for eqw (scratchpad/eqw/eqw.log) to write its anchored terminal marker (ANY DONE_EXIT_:
REM muongates needs a free GPU, not an eqw success), lets auto_control.py apply eqw's both-modes gate
REM and regenerate muongates' runner on the right base, then runs it. Refuses to launch if
REM auto_control.py fails. Polls every 2 min. Writes wait_muongates.log. CRLF -- see the goto rule.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\muongates\wait_muongates.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eqw\eqw.log
echo ===== muongates waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waiteqw
if not exist "%G0%" (
  ping -n 121 127.0.0.1 >nul
  goto waiteqw
)
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waiteqw
)
echo eqw reported, choosing the base %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/muongates/auto_control.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo auto_control FAILED -- not launching muongates %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_65 %DATE% %TIME% >> "%LOG%"
  exit /b 65
)
echo launching muongates %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\muongates\run_muongates.cmd
echo run_muongates returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
