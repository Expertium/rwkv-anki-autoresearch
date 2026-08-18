@echo off
REM Wait for iter 57 (last in the chain), then run iter 54's owed DECAY + EVAL.
REM Chain after the 2026-08-18 outage:  iter54 WS -- iter52 -- iter55 -- iter57 -- iter54 PHASE 2
REM
REM WHY iter 54 IS SPLIT: its resumed WS completed (i54_ws_10935.pth, 12:44) but the decay phase
REM never executed, while its KD guards passed by reading a STALE log left by an aborted attempt.
REM Full account in mk54_phase2.py. Queued LAST rather than inserted mid-chain, because iter 55's
REM waiter already polls iter52.log -- a second waiter on the same log would fire simultaneously
REM and put two jobs on one GPU.
REM
REM ANCHORED findstr /B: the unanchored form matches any line that merely MENTIONS the token,
REM including a waiter's own progress message, and fires the loop instantly.
REM Verified before arming: iter57.log does not yet exist, so nothing stale can satisfy this.
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
