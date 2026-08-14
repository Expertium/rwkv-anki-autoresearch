@echo off
REM Wait for the imm-independence chain to finish, then run the norm probe. Keeps the GPU busy
REM unattended; both are short.
REM
REM wait_immcorr.log is truncated (">") at each launch of its own chain, so its terminal token is
REM unambiguous in time here -- unlike cblearn.log, which is appended and needed artifact-based
REM waiting instead.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\wait_immcorr.log
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\wait_normprobe.log
echo waiting for the immcorr chain %DATE% %TIME% > "%LOG%"

:wait
findstr /B /C:"IMMCORRCHAIN_EXIT_" "%WLOG%" >nul
if %ERRORLEVEL%==0 goto ready
ping -n 61 127.0.0.1 >nul
goto wait

:ready
ping -n 31 127.0.0.1 >nul
echo immcorr done, starting the norm probe %DATE% %TIME% >> "%LOG%"
call scratchpad\qat_tax\run_normprobe.cmd
echo norm probe returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo NORMCHAIN_EXIT_0 %DATE% %TIME% >> "%LOG%"
