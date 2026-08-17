@echo off
REM Wait for iter 52's RE-RUN (the tail of the chain: 53 -> 54 -> 52 -> 55), then launch iter 55.
REM Detached via scratchpad/detach.ps1 so Esc / session teardown cannot kill it.
REM
REM Anchored findstr /B: the unanchored form matches any line that merely MENTIONS the token,
REM including a waiter's own progress message, which fires the loop instantly.
REM
REM iter 52's ORIGINAL log was renamed to iter52_failed_smoke_2135.log after its 21:35 launch died
REM in 0.07 s, precisely so this waiter cannot fire on that run's stale DONE_EXIT_45.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\iter52.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter55_rgate\waiter.log
echo waiter armed (waits on iter 52 re-run) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo iter 52 finished, launching iter 55 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter55_rgate\run_iter55.cmd
echo iter55 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
