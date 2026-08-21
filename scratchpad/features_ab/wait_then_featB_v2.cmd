@echo off
REM Launch featB (the FIXED CONTROL) once the id-fixed published TRAIN db exists and verifies.
REM Gated on the train build's own success marker, not merely on the directory existing --
REM data_processing is resumable, so a half-built db looks complete to an `if exist` check.
REM Anchored grep; this file never spells the token outside the check and its own marker.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set RB=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featA2\featA2.log
set NEXT=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featB\run_featB.cmd
set WLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\wait_featB.log
echo waiter armed %DATE% %TIME% -- polling featA2 for its TERMINAL MARKER > "%WLOG%"
:loop
findstr /B /C:"DONE_EXIT_" "%RB%" >nul 2>&1
if %ERRORLEVEL%==0 goto checks
timeout /t 60 /nobreak >nul
goto loop
:checks
findstr /B /C:"DONE_EXIT_0 " "%RB%" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo ABORT featA2 did not finish cleanly %DATE% %TIME% >> "%WLOG%"
  exit /b 22
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/test_db_5k_id3 2000 46 >> "%WLOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo ABORT gen-3 eval db failed check_db %DATE% %TIME% >> "%WLOG%"
  exit /b 23
)
echo launching featB %DATE% %TIME% >> "%WLOG%"
call "%NEXT%"
echo featB returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
