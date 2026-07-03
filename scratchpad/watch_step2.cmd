@echo off
REM Watcher: polls build_5k.log for the [STEP2_EXIT_0] marker (test_db 5001-10000 done), then
REM launches the d=128 baseline eval (run_base5k_eval.cmd). Detached via detach.ps1 -> survives Esc.
REM Nonzero STEP2 exit -> logs and stops WITHOUT launching. Poll = 5 min (ping-sleep; timeout /t
REM breaks without stdin in detached processes). Log: scratchpad/watch_step2.log
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\watch_step2.log
echo WATCH_START %DATE% %TIME% >> "%LOG%"

:loop
findstr /c:"[STEP2_EXIT_0]" scratchpad\build_5k.log >nul 2>&1
if %ERRORLEVEL%==0 goto launch
findstr /c:"[STEP2_EXIT_" scratchpad\build_5k.log >nul 2>&1
if %ERRORLEVEL%==0 goto failed
ping -n 301 127.0.0.1 >nul
goto loop

:launch
echo STEP2_DONE - launching baseline eval %DATE% %TIME% >> "%LOG%"
call scratchpad\run_base5k_eval.cmd
echo WATCH_DONE %DATE% %TIME% >> "%LOG%"
exit /b 0

:failed
echo STEP2_FAILED (nonzero exit marker in build_5k.log) - eval NOT launched %DATE% %TIME% >> "%LOG%"
exit /b 1
