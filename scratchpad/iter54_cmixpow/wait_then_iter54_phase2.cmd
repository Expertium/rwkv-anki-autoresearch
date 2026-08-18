@echo off
REM Wait for iter 52, then run iter 54's owed DECAY + EVAL.
REM Chain (REORDERED 2026-08-18 15:2x):  iter54 WS -- iter52 -- iter54 PHASE 2 -- iter55 -- iter57
REM
REM WHY iter 54 IS SPLIT: its resumed WS completed (i54_ws_10935.pth, 12:44) but the decay phase
REM never executed, while its KD guards passed by reading a STALE log left by an aborted attempt.
REM Full account in mk54_phase2.py. Originally queued LAST, MOVED TO SECOND 2026-08-18: iter 54 is
REM the oldest incomplete work and its owed decay+eval is only ~6.1 h, so finishing it a day later
REM left a half-done iteration exposed to another outage for nothing. The collision that first
REM forced it last -- iter 55's waiter also polled iter52.log, so both would have fired at once --
REM is resolved by re-pointing iter 55 at iter54_phase2.log, a FRESH file. iter54.log could NOT be
REM used for that: it already carries DONE_EXIT_TOMLFAIL_1 from the 12:44 failure.
REM
REM ANCHORED findstr /B: the unanchored form matches any line that merely MENTIONS the token,
REM including a waiter's own progress message, and fires the loop instantly.
REM Verified before arming: iter52.log carries 0 anchored markers (it is mid-decay).
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\iter52.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\waiter_phase2.log
echo waiter armed (waits on iter 52) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo iter 52 finished, launching iter 54 phase 2 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\run_iter54_phase2.cmd
echo iter54 phase2 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
