@echo off
REM Waits for the shared 10-epoch WS run to SUCCEED, then launches endgame arm 1 (the plain 2-epoch
REM decay + eval). Gates on the SUCCESS line, not merely a terminal marker: ws10 auto-resumes across
REM crashes and can still exit non-zero (no checkpoint, wrong param count, too many attempts), and a
REM failure marker in a shared log is a live trigger for everything downstream (CLAUDE.md).
REM Polls every 5 min -- the WS phase is ~34 h, so a tight poll buys nothing. CRLF: see the goto rule.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10plain\wait_w10plain.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ws10\ws10.log
echo ===== w10plain waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitws
if not exist "%G0%" (
  ping -n 301 127.0.0.1 >nul
  goto waitws
)
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 301 127.0.0.1 >nul
  goto waitws
)
findstr /B /C:"DONE_EXIT_0 " "%G0%" >nul 2>&1
if errorlevel 1 (
  echo ws10 ended WITHOUT success -- not launching arm 1 %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_65 %DATE% %TIME% >> "%LOG%"
  exit /b 65
)
if not exist "C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ws10\w10_ws_109350.pth" (
  echo ws10 reported success but the final checkpoint is missing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_66 %DATE% %TIME% >> "%LOG%"
  exit /b 66
)
echo ws10 succeeded, launching arm 1 %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10plain\run_w10plain.cmd
echo run_w10plain returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
