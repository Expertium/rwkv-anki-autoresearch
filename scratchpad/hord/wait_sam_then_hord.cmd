@echo off
REM Waits for sam (scratchpad/sam/sam.log) to write its anchored terminal marker (ANY DONE_EXIT_: hord
REM needs a free GPU, not a sam success), lets auto_control.py apply the both-modes gate to sam and
REM regenerate hord's runner accordingly (SAM in the decay iff sam promoted), then runs hord.
REM Refuses to launch if auto_control.py fails. Polls every 2 min. Writes wait_hord.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hord\wait_hord.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sam\sam.log
echo ===== hord waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitsam
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitsam
)
echo sam reported, choosing the base %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/hord/auto_control.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo auto_control FAILED -- not launching hord %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_65 %DATE% %TIME% >> "%LOG%"
  exit /b 65
)
echo launching hord %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hord\run_hord.cmd
echo run_hord returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
