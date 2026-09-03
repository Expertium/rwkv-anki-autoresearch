@echo off
REM Launch the LOO sweep when lorawd's chain has REPORTED (any terminal marker): lorawd is the last GPU
REM job of the features chain, so its marker means the GPU is free. run_loo.cmd may be
REM regenerated (python scratchpad/feat_loo/mk_loo.py) any time before this fires -- a called .cmd is not open
REM until the call. Anchored findstr. No angle brackets or arrows in REM lines.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_loo\wait_loo.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\lorawd\wait_lorawd.log

echo ===== loo waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo lorawd chain reported %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_loo\run_loo.cmd
echo run_loo returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
