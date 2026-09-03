@echo off
REM lorawd waiter v2 (2026-09-03 08:15): the v1 waiter polled wait_realcyc.log, which now carries the v2
REM refusal marker, and was killed before it could fire. Polls wait_realcyc3.log instead; still writes
REM wait_lorawd.log so the LOO waiter behind it stays valid. If realcyc promotes, run mk_lorawd.py realcyc
REM before this fires. No percent-tilde in REM lines.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\lorawd\wait_lorawd.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\wait_realcyc3.log

echo ===== lorawd waiter v2 armed %DATE% %TIME% ===== >> "%LOG%"

:waitall
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitall
)
echo all gates open %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\lorawd\run_lorawd.cmd
echo run_lorawd returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
