@echo off
REM Wait for the running QAT#2 chain to finish, then launch iter 52. Detached so it survives Esc.
REM
REM ANCHORED grep on purpose. `findstr /C:"DONE_EXIT"` also matches a log line that merely MENTIONS
REM the token -- including a waiter's own progress message -- which fires the loop instantly. Every
REM terminal line starts with the token, prose never does, so /B is the fix. Cost one wrongly
REM started co-tenant eval on 2026-07-26.
REM
REM Serialising matters: both jobs want the whole GPU, and running a gate-critical eval beside
REM another CUDA process is what deadlocked WDDM paging for 2.7 h once.
set QATLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\i45kd.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\waiter.log
echo waiter armed %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%QATLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 120 /nobreak >nul 2>&1
goto loop
:go
echo QAT chain finished, launching iter 52 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\run_iter52.cmd
echo iter52 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
