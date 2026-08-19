@echo off
REM =========================================================================================
REM Wait for rgate (the last queued research run), then start the BUG HUNT automatically.
REM
REM WHY THIS EXISTS: rgate is the end of the chain, so without this the GPU idles from the
REM moment it finishes until a human or agent is next active. Andrew's directive is that the
REM bug hunt begins when the GPU frees, and its first step is the `base` diagnostic -- nothing
REM else in the matrix is interpretable until that one is green.
REM
REM SAFE TO RUN UNATTENDED: the diagnostic is ~25 min, writes only into
REM scratchpad/bughunt/diag_base/, and its result jsonls are tagged RWKV-diag_base so they
REM cannot collide with any research run's. Its logloss is MEANINGLESS by construction and
REM must never be recorded as a result -- it exists to execute the path.
REM
REM ANCHORED findstr /B: the unanchored form matches any line that merely MENTIONS the token,
REM including a waiter's own progress message, and fires the loop instantly.
REM Verified before arming: iter55.log does not yet exist, so nothing stale can satisfy this.
REM =========================================================================================
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter55_rgate\iter55.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\bughunt\waiter_diag.log
echo waiter armed (waits on rgate, then starts the bug hunt) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo rgate finished, launching the base diagnostic %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\bughunt\diag_base\run_diag_base.cmd
echo diag_base returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
