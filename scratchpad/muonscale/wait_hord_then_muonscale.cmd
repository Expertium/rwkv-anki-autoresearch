@echo off
REM Waits for hord (scratchpad/hord/hord.log) to write its anchored terminal marker (ANY DONE_EXIT_:
REM muonscale needs a free GPU, not a hord success), lets auto_control.py apply hord's curve-side gate
REM and regenerate muonscale's runner on the right base, then runs it. Refuses to launch if
REM auto_control.py fails. Polls every 2 min. Writes wait_muonscale.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\muonscale\wait_muonscale.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hord\hord.log
echo ===== muonscale waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waithord
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waithord
)
echo hord reported, choosing the base %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/muonscale/auto_control.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo auto_control FAILED -- not launching muonscale %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_65 %DATE% %TIME% >> "%LOG%"
  exit /b 65
)
echo launching muonscale %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\muonscale\run_muonscale.cmd
echo run_muonscale returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
