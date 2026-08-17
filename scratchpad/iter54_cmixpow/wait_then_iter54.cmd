@echo off
REM Wait for iter 53, then launch iter 54. Detached. Anchored findstr (/B): the unanchored form
REM matches any line that merely MENTIONS the token, including a waiter's own progress message.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter53_muonlora\iter53.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\waiter.log
echo waiter armed %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo iter 53 finished, launching iter 54 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\run_iter54.cmd
echo iter54 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
