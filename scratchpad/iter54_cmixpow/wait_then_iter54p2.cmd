@echo off
REM Wait for iter 57, then run iter 54 PHASE 2 (its decay + eval). Chain as of 2026-08-18 13:15:
REM   iter52 (running) -- iter55 -- iter57 -- iter54_phase2
REM
REM iter 54's WS is COMPLETE on disk (i54_ws_10935.pth, 12:44); only its decay and eval are owed,
REM because the decay phase never executed while its KD guards passed on a stale log. Queued LAST
REM rather than next, because iter55's waiter already polls iter52.log and adding a second waiter
REM on the same log would start two GPU jobs at once.
REM
REM ANCHORED findstr /B: the unanchored form matches any line that merely MENTIONS the token,
REM including this waiter's own progress message, and fires the loop instantly.
REM Verified before arming: iter57.log does not yet exist, so there is no stale marker to trip on.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter57_decayshape\iter57.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\waiter_phase2.log
echo waiter armed (waits on iter 57) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo iter 57 finished, launching iter 54 phase 2 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\run_iter54_phase2.cmd
echo iter54 phase2 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
