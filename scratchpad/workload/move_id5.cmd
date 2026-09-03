@echo off
REM Move the gen-5 TRAIN db from F: (USB HDD, random-read-bound at ~11.5 MB/s) to C: (SSD), then
REM swap the F: path for a junction so no toml or guard string changes. Andrew authorized deleting
REM train_db_5k_h1_id3 on 2026-09-03 to make the room. Two scripts by design: the verifier's own
REM handle blocks the rename if done in one process (CLAUDE.md, the e2s move).
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\workload\move_id5.log
echo ===== MOVE train_db_5k_h1_id5 F to C START %DATE% %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/workload/move_lmdb.py train_db_5k_h1_id5 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo MOVE_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_11 %DATE% %TIME% >> "%LOG%"
  exit /b 11
)
findstr /C:"VERIFY OK" "%LOG%" >nul
if not %ERRORLEVEL%==0 (
  echo NO VERIFY OK LINE -- not finalizing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_12 %DATE% %TIME% >> "%LOG%"
  exit /b 12
)
.venv\Scripts\python.exe -u scratchpad/workload/finalize_lmdb.py train_db_5k_h1_id5 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo FINALIZE_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_13 %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
echo MOVE_OK %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
