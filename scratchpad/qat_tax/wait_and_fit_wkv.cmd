@echo off
REM Wait for the WKV corpus dump to finish, then fit the d=80 catalog. CPU-only both.
REM
REM ⚠ WAITLOOP TRAP: findstr /C:"TOKEN" also matches a log line that merely MENTIONS the token --
REM including a waiter's own progress message -- which fires the loop instantly. Anchored with /B
REM (terminal lines start with the token, prose never does), and this file deliberately never
REM echoes the awaited token anywhere.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DUMPLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\dump_wkv.log
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\wait_fit.log
echo waiting for the corpus dump %DATE% %TIME% > "%LOG%"

:wait
findstr /B /C:"DUMP_DONE_EXIT_" "%DUMPLOG%" >nul
if %ERRORLEVEL%==0 goto ready
REM ~15 s poll without needing a sleep binary
ping -n 16 127.0.0.1 >nul
goto wait

:ready
echo dump finished, fitting %DATE% %TIME% >> "%LOG%"
call scratchpad\qat_tax\fit_wkv_cb.cmd
echo fit returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo CHAINDONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
