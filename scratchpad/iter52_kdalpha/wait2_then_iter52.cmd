@echo off
REM RE-QUEUE of iter 52 after its 2026-08-17 21:35 launch died in 0.07 s.
REM
REM WHY IT DIED (both faults now fixed in run_iter52.cmd, and both were in phase 0 only):
REM   1. the guard called "Git\usr\bin\bash.exe" -- the RAW MSYS binary, which from cmd.exe has no
REM      MSYS PATH, so the script's cd "$(dirname "$0")/../.." died on 'dirname: command not found'.
REM      "Git\bin\bash.exe" is the wrapper that sets that PATH up. Both were reproduced through
REM      cmd.exe before the change.
REM   2. smoke_scripted_eval.sh takes a REQUIRED eval_toml argument and was called with none. It
REM      now gets iter 45's, which points at a real checkpoint -- the right choice regardless,
REM      since the guard tests whether the CURRENT CODE scripts and runs, not iter 52's weights.
REM
REM Now waits on ITER 54 (the current tail of the chain), not on the QAT log it originally polled.
REM Anchored findstr /B: the unanchored form matches any line that merely MENTIONS the token,
REM including a waiter's own progress message, and fires the loop instantly.
REM
REM The previous run's log was renamed to iter52_failed_smoke_2135.log on purpose -- it still
REM contains a DONE_EXIT_45 line, and leaving it in place would make any waiter on iter52.log fire
REM immediately. run_iter52.cmd appends, so a stale terminal line is not self-clearing.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\iter54.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\waiter2.log
echo waiter2 armed (waits on iter 54) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo iter 54 finished, launching iter 52 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\run_iter52.cmd
echo iter52 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
