@echo off
REM ===========================================================================================
REM Move BOTH e2s dbs to the SSD, then start the re-base.
REM
REM Andrew authorised deleting train_db_5k_h1, train_db_5k_h1_fix and test_db_5k_fix
REM (2026-08-30), freeing 297.4 GB on C:. Measured penalty for reading a db from the USB drive
REM is 2.2x per step -- the original C:-hosted teacher dump ran at 1.40 steps/s, the same dump
REM on F: managed 0.63, and GPU utilisation during it was 8%, i.e. starved on reads.
REM
REM ---- WHY BOTH MOVES HAPPEN BEFORE THE RE-BASE, AND NOT DURING IT ----
REM Copying the test db during the WS phase looked like free parallelism: WS reads the TRAIN db,
REM which by then is on C:, so the F: read would not contend. But the FINALIZE step renames the
REM original and drops a junction in its place, and ws.toml sets VALIDATE_EVERY = 1000 with
REM VALIDATE_DATASET_LMDB_PATH pointing at that very test db -- so WS opens it periodically. A
REM rename against an open LMDB fails with a bare Access Denied, which is precisely the failure
REM already recorded when an orphan fetch worker held a db open for three days.
REM So: both moves complete first. It costs about 2 h of idle GPU once, and buys a fast eval on
REM this run and on every future one, with no window in which a rename can collide.
REM
REM Two tools, not one, and deliberately: move_lmdb.py verifies by opening an LMDB env on the
REM source and then cannot rename it, because its own handle is still associated with the
REM directory. finalize_lmdb.py never opens the source.
REM
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\workload\move_both.log
set MOVELOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\workload\move_e2s.log
set WORK=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\workload

echo ===== move-both chain armed %DATE% %TIME% ===== >> "%LOG%"

REM ---- 1. the TRAIN copy is already running; wait for its verify ----
:waitcopy
findstr /B /C:"VERIFY OK" "%MOVELOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 61 127.0.0.1 >nul
  goto waitcopy
)
echo train copy verified %DATE% %TIME% >> "%LOG%"

REM second witness: the copying process must be gone before anything is renamed
:waitproc
wmic process where "name='python.exe'" get commandline 2>nul | findstr /C:"move_lmdb" >nul 2>&1
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto waitproc
)
echo train copy process gone %DATE% %TIME% >> "%LOG%"

.venv\Scripts\python.exe scratchpad/workload/finalize_lmdb.py train_db_5k_h1_e2s >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo TRAIN_FINALIZE_FAILED -- the tool restores the original on any failure >> "%LOG%"
  echo DONE_EXIT_51 %DATE% %TIME% >> "%LOG%"
  exit /b 51
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/train_db_5k_h1_e2s 1483984 24 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo TRAIN_JUNCTION_CHECK_FAILED >> "%LOG%"
  echo DONE_EXIT_52 %DATE% %TIME% >> "%LOG%"
  exit /b 52
)
echo TRAIN db now reads from the SSD %DATE% %TIME% >> "%LOG%"

REM ---- 2. the TEST db ----
echo --- copying the test db %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/workload/move_lmdb.py test_db_5k_e2s >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo TEST_COPY_FAILED -- both copies left in place, nothing renamed >> "%LOG%"
  echo DONE_EXIT_53 %DATE% %TIME% >> "%LOG%"
  exit /b 53
)
.venv\Scripts\python.exe scratchpad/workload/finalize_lmdb.py test_db_5k_e2s >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo TEST_FINALIZE_FAILED >> "%LOG%"
  echo DONE_EXIT_54 %DATE% %TIME% >> "%LOG%"
  exit /b 54
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/test_db_5k_e2s 170384 24 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo TEST_JUNCTION_CHECK_FAILED >> "%LOG%"
  echo DONE_EXIT_55 %DATE% %TIME% >> "%LOG%"
  exit /b 55
)
echo TEST db now reads from the SSD %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\e2s_rebase\run_e2s_rebase.cmd
echo re-base returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
