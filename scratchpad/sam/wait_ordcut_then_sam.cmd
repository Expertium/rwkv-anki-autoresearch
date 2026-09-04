@echo off
REM Waits for ordcut (scratchpad/ordcut/ordcut.log) to write its anchored terminal marker (ANY
REM DONE_EXIT_: sam needs a free GPU, not an ordcut success), lets auto_control.py pick the base
REM (ordcut if its curve-side gate passed, else ordcut's own control) and regenerate + preflight the
REM decay-only runner, then runs it. Refuses to launch if auto_control.py fails. Polls every 2 min.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sam\wait_sam.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ordcut\ordcut.log
echo ===== sam waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitoc
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitoc
)
echo ordcut reported, choosing the base %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/sam/auto_control.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo auto_control FAILED -- not launching sam %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_65 %DATE% %TIME% >> "%LOG%"
  exit /b 65
)
echo launching sam %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sam\run_sam.cmd
echo run_sam returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
