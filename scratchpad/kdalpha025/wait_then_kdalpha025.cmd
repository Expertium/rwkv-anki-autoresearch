@echo off
REM Wait for the decayshape run (currently last), then run KD alpha_decay 0.25.
REM Chain: iter54 phase 2 -- rgate -- decayshape -- kdalpha025
REM
REM ANCHORED findstr /B: the unanchored form matches any line that merely MENTIONS the token,
REM including a waiter's own progress message, and fires the loop instantly.
REM Verified before arming: iter57.log does not exist yet, so nothing stale can satisfy this.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter57_decayshape\iter57.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\kdalpha025\waiter.log
echo waiter armed (waits on decayshape) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo decayshape finished, launching kdalpha025 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\kdalpha025\run_kdalpha025.cmd
echo kdalpha025 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
