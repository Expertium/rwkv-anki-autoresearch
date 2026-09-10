@echo off
REM Arm 2's phase-C launcher. run_w10qat.cmd's cmd was stopped on 2026-09-10 so that its phase C
REM could not evaluate with the START catalogs; its decay python kept running. This waiter polls
REM decay_state.py about once a minute:
REM   rc 10 = the decay python is alive: keep waiting
REM   rc 0  = it has exited AND the step-21870 checkpoint and both learned catalogs exist: launch
REM   rc 20 = it has exited WITHOUT them: refuse, loudly, and launch nothing
REM Any other rc is a failure of the check itself: retried 5 times, then refused.
REM decay_state.txt is overwritten on every poll, so its mtime is this waiter's heartbeat.
REM This log carries the WAITER's own DONE_EXIT_ lines (the repo convention) and NOTHING polls it:
REM the downstream waiters poll w10qat_eval.log, which only run_w10qat_eval.cmd writes.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\wait_qateval.log
set STATE=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\decay_state.txt
set BAD=0
echo ===== arm-2 eval waiter armed %DATE% %TIME% ===== >> "%WLOG%"

:poll
.venv\Scripts\python.exe scratchpad\w10qat\decay_state.py > "%STATE%" 2>&1
set RC=%ERRORLEVEL%
if "%RC%"=="10" goto sleep
if "%RC%"=="0" goto finished
if "%RC%"=="20" goto dead
set /a BAD=BAD+1
echo state check failed rc=%RC% attempt %BAD% %DATE% %TIME% >> "%WLOG%"
type "%STATE%" >> "%WLOG%"
if %BAD% GEQ 5 goto broken
:sleep
ping -n 61 127.0.0.1 >nul
goto poll

:dead
echo decay python EXITED WITHOUT its step-21870 artifacts -- not launching the eval %DATE% %TIME% >> "%WLOG%"
type "%STATE%" >> "%WLOG%"
echo DONE_EXIT_20 %DATE% %TIME% >> "%WLOG%"
exit /b 20

:broken
echo the state check itself keeps failing -- not launching the eval %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_21 %DATE% %TIME% >> "%WLOG%"
exit /b 21

:finished
type "%STATE%" >> "%WLOG%"
echo decay finished -- 120 s for its fetch workers to release the GPU, then the eval %DATE% %TIME% >> "%WLOG%"
ping -n 121 127.0.0.1 >nul
echo launching run_w10qat_eval.cmd %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\run_w10qat_eval.cmd
echo run_w10qat_eval.cmd returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
