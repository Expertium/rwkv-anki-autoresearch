@echo off
REM Wait for the decayshape run, then run iter 54's owed DECAY + EVAL (attempt b).
REM Chain: decayshape -- iter54 phase 2b -- kdalpha025
REM
REM Phase 2a decayed for 3.3 h at KD alpha 0.9 instead of 0.5, because the generator sliced away
REM the WS region that contains the reset line. Its own KD guard caught it and exited
REM DONE_EXIT_WRONGALPHA_DECAY; the bad checkpoint is quarantined as i54_d_10935.WRONGALPHA09.pth.
REM
REM Polls iter57.log. It must NOT poll iter54_phase2.log (phase 2a's failure marker) and must NOT
REM poll iter55.log, which carries rgate's DONE_EXIT_46 smoke failure.
REM ANCHORED findstr /B: the unanchored form matches any line that merely MENTIONS the token.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter57_decayshape\iter57.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\waiter_phase2b.log
echo waiter armed (waits on decayshape) %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 300 /nobreak >nul 2>&1
goto loop
:go
echo decayshape finished, launching iter 54 phase 2b %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter54_cmixpow\run_iter54_phase2b.cmd
echo phase2b returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
