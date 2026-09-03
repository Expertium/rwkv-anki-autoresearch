@echo off
REM realcyc waiter v3 (2026-09-03 08:15): v2 refused at 08:07 because rebuild5.log carried a failure
REM marker from the RAM-starved test build and the eval resume had failed again. Gates now key on SUCCESS
REM markers: the eval resume's gen4base EVAL_OK line (a DONE_EXIT_ there is stale from the failed attempts)
REM and rebuild5.log DONE_EXIT_0 (refuse only on DONE_EXIT_15 appearing WITHOUT a later DONE_EXIT_0 -- the
REM restart appends to the same log, so the failure line stays; findstr sees both). Writes wait_realcyc3.log,
REM a fresh file, because wait_realcyc.log already carries DONE_EXIT_62. No percent-tilde in REM lines.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\wait_realcyc3.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\gen4base_evalresume.log
set G1=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\rebuild5.log

echo ===== realcyc waiter v3 armed %DATE% %TIME% ===== >> "%LOG%"

:waitall
findstr /B /C:"gen4base EVAL_OK" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitall
)
findstr /B /C:"DONE_EXIT_0" "%G1%" >nul 2>&1
if errorlevel 1 (
  findstr /B /C:"DONE_EXIT_15" "%G1%" >nul 2>&1
  if not errorlevel 1 (
    echo gate 1 FAILED: DONE_EXIT_15 present, DONE_EXIT_0 absent -- refusing %DATE% %TIME% >> "%LOG%"
    echo DONE_EXIT_62 %DATE% %TIME% >> "%LOG%"
    exit /b 62
  )
  ping -n 121 127.0.0.1 >nul
  goto waitall
)
echo all gates open %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\run_realcyc.cmd
echo run_realcyc returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
