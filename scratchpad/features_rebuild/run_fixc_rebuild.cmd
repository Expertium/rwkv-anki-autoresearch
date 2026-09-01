@echo off
REM ===========================================================================================
REM THE END-TO-END CONTROL REBUILD (`fixc`) -- built with CURRENT code so it pairs with the e2s
REM dbs and isolates the interval definition.
REM
REM WHY IT IS NEEDED. `train_db_5k_h1_fix` (featA2's db) was built 2026-08-21, BEFORE the Bug C
REM fix; the e2s dbs were built 2026-08-30, WITH it. Measured with diff_db_ids.py: note_id
REM differs on 45.9% of entries, distinct 32,747 versus 57,971, while card/deck/preset/user are
REM byte-identical. So an experiment spanning those two dbs measures the interval change PLUS a
REM note-identity fix whose sibling (Bug A) was independently worth +0.000148 / +0.000169.
REM Andrew approved the spend 2026-08-30.
REM
REM ⚠ THE ONE THING THAT MUST NOT HAPPEN: RWKV_E2S_PUBLISHED leaking in and making this "control"
REM a second copy of the treatment. That is the rgate false-green shape -- a control arm that
REM inherited its treatment from the ambient environment, whose inertness check then passed
REM VACUOUSLY at 0.000e+00 while comparing two treated models. So the variable is REFUSED if set,
REM cleared explicitly rather than assumed absent, AND the final phase positively requires the
REM features to DIFFER from the e2s db.
REM
REM CPU-only, so it runs alongside the queued GPU work. Cost from the e2s build: test 50 min,
REM train 18 min. label_filter_db is REUSED: it selects WHICH reviews count, not what they hold.
REM
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild
set LOG=%DIR%\fixc_rebuild.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set RWKV_ID_FEATURES=

echo ===== FIXC REBUILD START %DATE% %TIME% ===== >> "%LOG%"

REM ---- phase 0: refuse to build a control with the treatment's lever set ----
REM `if defined` rather than a percent-expanded comparison: it tests the INHERITED environment
REM without expanding an as-yet-unset name, which is both the idiomatic cmd form and the one that
REM does not read as a use-before-set to a static checker. Do not write percent-wrapped names in
REM a REM line either -- the checker reads comments too, and cmd itself parses redirection there.
if defined RWKV_E2S_PUBLISHED (
  echo E2S_LEVER_SET -- refusing to build the end-to-end control >> "%LOG%"
  echo DONE_EXIT_44 >> "%LOG%"
  exit /b 44
)
REM cleared explicitly, not merely absent -- the child process inherits this environment
set RWKV_E2S_PUBLISHED=
echo FIXC LEVER_OFF_CONFIRMED %TIME% >> "%LOG%"

REM ---- phase A: the TEST db ----
echo --- test db %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_test_5k_fixc.toml > "%DIR%\fixc_test_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo FIXC TESTFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_21 >> "%LOG%"
  exit /b 21
)
echo FIXC TEST_OK %TIME% >> "%LOG%"

REM ---- phase A2: prove the TEST pair is single-variable before spending an hour on the train db
.venv\Scripts\python.exe scratchpad/features_rebuild/assert_pair_single_variable.py F:/rwkv_lmdb/test_db_5k_fixc F:/rwkv_lmdb/test_db_5k_e2s 40 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo FIXC TEST_PAIR_INVALID %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_45 >> "%LOG%"
  exit /b 45
)
echo FIXC TEST_PAIR_VALID %TIME% >> "%LOG%"

REM ---- phase B: the TRAIN db ----
echo --- train db %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_train_5k_h1_fixc.toml > "%DIR%\fixc_train_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo FIXC TRAINFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_22 >> "%LOG%"
  exit /b 22
)
echo FIXC TRAIN_OK %TIME% >> "%LOG%"

REM ---- phase C: width, then the train pair ----
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/train_db_5k_h1_fixc 100000 24 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo FIXC CHECKDB_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_23 >> "%LOG%"
  exit /b 23
)
.venv\Scripts\python.exe scratchpad/features_rebuild/assert_pair_single_variable.py F:/rwkv_lmdb/train_db_5k_h1_fixc F:/rwkv_lmdb/train_db_5k_h1_e2s 40 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo FIXC TRAIN_PAIR_INVALID %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_46 >> "%LOG%"
  exit /b 46
)
echo FIXC TRAIN_PAIR_VALID %TIME% >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
