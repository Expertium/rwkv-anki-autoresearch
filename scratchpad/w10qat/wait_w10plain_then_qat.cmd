@echo off
REM Waits for ENDGAME ARM 1 to SUCCEED, then refits the PQ catalogs on the ws10 WS-final and runs
REM ARM 2. One waiter, strictly sequential, so the CPU-heavy k-means never competes with a
REM training run for fetch workers -- it costs ~15 min of idle GPU and buys no contention.
REM
REM Gates on the SUCCESS line, never on a bare terminal marker: a failure marker in a shared log
REM is a live trigger for everything downstream (CLAUDE.md), and arm 1 can exit non-zero from any
REM of its own phase guards.
REM
REM Why the catalog refit is here and not a note: both deployed catalogs were measured STALE on
REM the gen-5 trunk (the shift one scored WORSE than encoding to zero), a stale catalog passes
REM every shape assert, and arm 2 exists to price quantization. Arm 2's phase 0 refuses without
REM the ws10-fitted pair, so this step is what unblocks it.
REM CRLF: see the goto rule.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\wait_qat.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10plain\w10plain.log
echo ===== arm-2 waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitarm1
if not exist "%G0%" (
  ping -n 301 127.0.0.1 >nul
  goto waitarm1
)
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 301 127.0.0.1 >nul
  goto waitarm1
)
findstr /B /C:"DONE_EXIT_0 " "%G0%" >nul 2>&1
if errorlevel 1 (
  echo arm 1 ended WITHOUT success -- not running arm 2 %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_71 %DATE% %TIME% >> "%LOG%"
  exit /b 71
)
echo arm 1 succeeded, refitting the catalogs %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\run_fit_catalogs.cmd
echo run_fit_catalogs returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
findstr /B /C:"DONE_EXIT_0 " "C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\fitcb.log" >nul 2>&1
if errorlevel 1 (
  echo catalog refit did not succeed -- arm 2 would start from a stale catalog %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_72 %DATE% %TIME% >> "%LOG%"
  exit /b 72
)
echo catalogs refitted, launching arm 2 %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\run_w10qat.cmd
echo run_w10qat returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
