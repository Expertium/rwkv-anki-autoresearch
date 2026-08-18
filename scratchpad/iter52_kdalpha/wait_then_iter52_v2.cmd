@echo off
REM Wait for iter 54's RESUMED run, then launch iter 52. Chain after the 2026-08-18 power outage:
REM   iter54 (resumed from step 8000) -- iter52 -- iter55 -- iter57
REM
REM ** WHY THIS FILE EXISTS instead of wait_then_iter52.cmd: that one polls the QAT#2 log
REM (scratchpad\qat_tax\i45kd.log), which ALREADY carries a terminal marker from 2026-08-17. Arming
REM it now would fire instantly and run iter 52 BESIDE iter 54 -- two processes contending for the
REM whole GPU, which is what deadlocked WDDM paging for 2.7 h once. Repointing is not enough to do
REM in place, because the old file is the record of how the pre-outage chain was wired.
REM
REM ANCHORED findstr /B on purpose: the unanchored form matches any line that merely MENTIONS the
REM token, including a waiter's own progress message, and fires the loop immediately.
REM Verified before arming: iter54.log contains NO anchored marker (all three of its run headers
REM are progress lines), so this waits rather than firing on the pre-outage history.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\iter54.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\waiter_v2.log
echo waiter armed (waits on the RESUMED iter 54) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo iter 54 finished, launching iter 52 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\run_iter52.cmd
echo iter52 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
