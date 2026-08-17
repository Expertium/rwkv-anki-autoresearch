@echo off
REM Wait for iter 55, then launch iter 57. Chain: 53 -> 54 -> 52 -> 55 -> 57.
REM Detached via scratchpad/detach.ps1 so Esc / session teardown cannot kill it.
REM Anchored findstr /B: the unanchored form matches any line that merely MENTIONS the token,
REM including a waiter's own progress message, and fires the loop instantly.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter55_rgate\iter55.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter57_decayshape\waiter.log
echo waiter armed (waits on iter 55) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo iter 55 finished, launching iter 57 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter57_decayshape\run_iter57.cmd
echo iter57 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
