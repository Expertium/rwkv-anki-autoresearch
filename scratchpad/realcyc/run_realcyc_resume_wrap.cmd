@echo off
REM Wrapper: call the resume runner, then write the two lines the (killed) v3 waiter would have
REM written to wait_realcyc3.log -- lorawd's waiter polls that file for an anchored DONE_EXIT_.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\wait_realcyc3.log
echo resume wrapper start %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\run_realcyc_resume.cmd
echo run_realcyc_resume returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
