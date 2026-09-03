@echo off
REM Waits for the gen-5 train db move (scratchpad/workload/move_id5.log) to report MOVE_OK, then
REM launches run_realcyc_resume_wrap.cmd (WS resume from rc_ws_3000 on the SSD -> decay -> eval;
REM the wrapper writes lorawd's trigger into wait_realcyc3.log). Gates on the SUCCESS line, not on
REM DONE_EXIT_: a failed move leaves DONE_EXIT_1x and this waiter must refuse, not fire.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\wait_move.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\workload\move_id5.log
echo ===== move waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitmove
findstr /B /C:"MOVE_OK" "%G0%" >nul 2>&1
if not errorlevel 1 goto go
findstr /B /C:"DONE_EXIT_1" "%G0%" >nul 2>&1
if not errorlevel 1 (
  echo move FAILED -- refusing to resume %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_63 %DATE% %TIME% >> "%LOG%"
  exit /b 63
)
ping -n 61 127.0.0.1 >nul
goto waitmove
:go
if not exist "F:\rwkv_lmdb\train_db_5k_h1_id5\data.mdb" (
  echo junction does not resolve -- refusing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_64 %DATE% %TIME% >> "%LOG%"
  exit /b 64
)
echo move OK, launching resume %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\run_realcyc_resume_wrap.cmd
echo resume wrap returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
