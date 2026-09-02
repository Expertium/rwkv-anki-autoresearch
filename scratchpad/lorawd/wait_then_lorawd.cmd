@echo off
REM Launch lorawd when realcyc's chain has REPORTED (any terminal marker): realcyc is the last GPU
REM job of the features chain, so its marker means the GPU is free. run_lorawd.cmd may be
REM regenerated (mk_lorawd.py realcyc) any time before this fires -- a called .cmd is not open
REM until the call. Anchored findstr. No angle brackets or arrows in REM lines.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\lorawd\wait_lorawd.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\wait_realcyc.log

echo ===== lorawd waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo realcyc chain reported %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\lorawd\run_lorawd.cmd
echo run_lorawd returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
