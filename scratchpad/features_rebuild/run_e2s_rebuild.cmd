@attrib +a "%~f0" >nul 2>&1
@echo off
REM ===========================================================================================
REM THE INTERVAL ARM: rebuild the PUBLISHED dbs with END-TO-START intervals.
REM
REM Andrew 2026-08-30: "let's do the interval comparison on the old dataset."
REM
REM WHY THE OLD DATASET. featB bundles TWO changes -- the new -id features AND end-to-start --
REM because the -id correction is gated on the dataset and fires automatically whenever
REM `review_time` is present. No -id database with end-to-end intervals exists, so that A/B
REM cannot separate them. On the PUBLISHED set the correction is one line
REM (elapsed_seconds - duration(k)/1000, both public columns), which isolates the interval
REM definition with the features held fixed.
REM
REM THE CONTROL ALREADY EXISTS AND COSTS NOTHING EXTRA: featA2 trained on train_db_5k_h1_fix,
REM the same generation, end-to-end. Its eval is phase 2 of the plan regardless.
REM
REM CPU-ONLY, so this runs alongside GPU work. OMP_NUM_THREADS=6 leaves headroom for a training
REM job's fetch workers.
REM
REM Cost, measured from the _fix rebuild (trainfix.log 21.08): test 30 min, train 50 min.
REM Output on F: -- C: has only 69 GB free.
REM label_filter_db is REUSED: it selects WHICH reviews count, not what they contain.
REM
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild
set LOG=%DIR%\e2s_rebuild.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set RWKV_ID_FEATURES=
REM ---- THE LEVER, and the only difference from the _fix build ----
set RWKV_E2S_PUBLISHED=1

echo ===== E2S REBUILD START %DATE% %TIME% ===== >> "%LOG%"

REM ---- phase 0: prove the lever is live before spending 80 minutes on it ----
.venv\Scripts\python.exe scratchpad/features_rebuild/smoke_e2s_published.py 4 > "%DIR%\e2s_smoke_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo E2S SMOKE_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_44 >> "%LOG%"
  exit /b 44
)
echo E2S SMOKE_OK %TIME% >> "%LOG%"

REM ---- phase A: the TEST db (eval feeds the same feature vector, so it must match) ----
echo --- test db %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_test_5k_e2s.toml > "%DIR%\e2s_test_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo E2S TESTFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_21 >> "%LOG%"
  exit /b 21
)
echo E2S TEST_OK %TIME% >> "%LOG%"

REM ---- phase B: the TRAIN db ----
echo --- train db %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_train_5k_h1_e2s.toml > "%DIR%\e2s_train_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo E2S TRAINFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_22 >> "%LOG%"
  exit /b 22
)
echo E2S TRAIN_OK %TIME% >> "%LOG%"

REM ---- phase C: the db must have the expected width and be readable ----
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/train_db_5k_h1_e2s 100000 24 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo E2S CHECKDB_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_23 >> "%LOG%"
  exit /b 23
)

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
