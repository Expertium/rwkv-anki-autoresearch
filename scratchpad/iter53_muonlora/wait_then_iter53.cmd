@echo off
REM Wait for iter 52 to finish, then launch iter 53. Detached so it survives Esc.
REM
REM ANCHORED grep (/B): the unanchored form matches any line that merely MENTIONS the token,
REM including a waiter's own progress message, and fires instantly.
REM
REM ** ON THE GATE BASIS, decided deliberately rather than by omission: iter 53 is built on the
REM ITER-45 recipe, so if iter 52 wins and promotes, iter 53's controlled comparison is still
REM iter-53-vs-iter-45 (single variable) while the CHAMPION it must beat has moved. That is
REM accepted on purpose -- the lever is orthogonal to KD alpha, its main value is testing the
REM regularizer prediction, and chaining keeps the GPU busy instead of idling until someone reads
REM iter 52's verdict. If iter 52 wins AND iter 53 shows signal, the stacked version is a cheap
REM follow-up rather than a lost run.
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter52_kdalpha\iter52.log
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter53_muonlora\waiter.log
echo waiter armed %DATE% %TIME% > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto go
timeout /t 180 /nobreak >nul 2>&1
goto loop
:go
echo iter 52 finished, launching iter 53 %DATE% %TIME% >> "%WLOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter53_muonlora\run_iter53.cmd
echo iter53 returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
