@echo off
REM Wait for kdalpha025, then RETRY rgate. Chain: decayshape -- iter54 phase2b -- kdalpha025 -- rgate
REM
REM Why rgate is being retried: its 22:25 launch failed its own pre-flight smoke, not the GPU.
REM run_iter55.cmd exports RWKV_RGATE=card BEFORE calling smoke_rgate.py, and the smoke built its
REM arms with dict(os.environ, **extra), so the OFF control inherited the flag. The param check
REM caught it; the inertness check had passed VACUOUSLY comparing two gated models. The smoke is
REM now hermetic (it strips its own vars per arm) and passes under the exact failing condition.
REM
REM ANCHORED findstr /B. iter55.log was renamed to iter55_failed_smoke_2225.log so the retry's
REM own marker cannot be shadowed by DONE_EXIT_46.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\kdalpha025\kdalpha025.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter55_rgate\waiter_retry.log
echo waiter armed (waits on kdalpha025) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo kdalpha025 finished, retrying rgate %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter55_rgate\run_iter55.cmd
echo rgate returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
