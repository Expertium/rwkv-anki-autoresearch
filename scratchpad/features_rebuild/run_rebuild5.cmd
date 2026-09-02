@echo off
REM =========================================================================================
REM THE FEATURES REBUILD, GENERATION 5 -- REAL-TIME CYCLES (RWKV_REAL_CYCLES=1).
REM Andrew 2026-09-02: "use real features for 3 days/week/month/year/decade/century, so that
REM every pseudo feature is replaced with its real counterpart. If it requires an LMDB rebuild,
REM ok." And: "11 is also a pseudo-calendar feature, so make sure it also gets replaced."
REM Gen 5 is gen 4 plus that single lever: the 28 pseudo day-offset cycle dims leave the
REM encoding block, the pseudo day_of_week column leaves the card features, and 24 real-time
REM cycle columns join the card-feature block (width 46 to 69, model input 114 to 109).
REM
REM   1. train_db_5k_h1_id5   users 1-5000
REM   2. test_db_5k_id5       users 5001-10000
REM
REM NO LABEL-FILTER PHASE. label_filter_db_id_e2s is REUSED: the equalize selection depends on
REM the interval definition, not on which feature columns exist, so gen 5 scores the SAME
REM reviews as gen 4. Phase 3 therefore REQUIRES identical counts -- the opposite of gen 4's
REM phase 3, whose new filter made a difference the expected outcome. Same guard, meaning
REM flipped by a config decision, revisited on purpose.
REM
REM PHASE 0 GUARDS: preflight, WIDTH (69 -- proves RWKV_REAL_CYCLES reached the process; a build
REM without it reproduces gen 4 under a gen-5 name and passes every count check), Bug C live,
REM targets. AND THE ARTIFACT IS CHECKED: the cycles checker reads each finished db back and
REM asserts the 24 cycle columns are unit-circle pairs with a shifted negative control.
REM
REM Built on F: beside gen 4. Nothing is deleted. Do NOT edit while running.
REM No angle brackets or arrows in REM lines.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild
set LOG=%DIR%\rebuild5.log
set STAMP=%RANDOM%%RANDOM%
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set WIDTH=69
set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
REM Must stay CLEARED: the published end-to-start formula subtracts the wrong review's duration
REM on an -id frame. data_processing asserts against it, but clear it rather than rely on that.
set RWKV_E2S_PUBLISHED=

if not exist "%DIR%" mkdir "%DIR%"
echo ===== FEATURES REBUILD GEN5 START %DATE% %TIME% ===== > "%LOG%"

REM ---- PHASE 0: everything the chain ASSUMES, checked before spending ~7 hours ----
REM Targets must not already exist (data_processing and find_equalize SKIP users already
REM present, so a pre-existing store reports success in seconds having done nothing); gen 3 must
REM still be there for phase 3 and because featB is scored against it; the -id dataset must be
REM readable; and F: must have the space, measured by FREE SPACE because these stores are sparse
REM and file length reports the map_size reservation instead of the allocation.
.venv\Scripts\python.exe scratchpad/features_rebuild/preflight_gen5.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 PREFLIGHT_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_50 %DATE% %TIME% >> "%LOG%"
  exit /b 50
)
echo REBUILD5 PREFLIGHT_OK %TIME% >> "%LOG%"

REM ---- PHASE 0a: the flag really produces WIDTH columns ----
.venv\Scripts\python.exe -c "import os,sys; sys.path.insert(0,os.getcwd()); import rwkv.id_features as f; w=f.card_feature_width(); print('card_feature_width', w, 'input_width', f.input_width()); sys.exit(0 if w==%WIDTH% else 44)" >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 WIDTH_MISMATCH %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_44 %DATE% %TIME% >> "%LOG%"
  exit /b 44
)
echo REBUILD5 WIDTH_OK %TIME% >> "%LOG%"

REM ---- PHASE 0b: the Bug C fix is LIVE, and the guard could tell if it were not ----
.venv\Scripts\python.exe scratchpad/features_rebuild/assert_bugc_fixed.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 BUGC_NOT_LIVE %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_45 %DATE% %TIME% >> "%LOG%"
  exit /b 45
)
echo REBUILD5 BUGC_GUARD_OK %TIME% >> "%LOG%"

REM ---- PHASE 0c: the configs point at _id5, so a stale clone cannot overwrite featB's gen 3 ----
findstr /C:"train_db_5k_h1_id5" rwkv\data_processing_train_5k_h1_id5.toml >nul
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TRAIN_TARGET_WRONG %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_49 %DATE% %TIME% >> "%LOG%"
  exit /b 49
)
findstr /C:"test_db_5k_id5" rwkv\data_processing_test_5k_h2_id5.toml >nul
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TEST_TARGET_WRONG %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_49 %DATE% %TIME% >> "%LOG%"
  exit /b 49
)
echo REBUILD5 TARGETS_OK %TIME% >> "%LOG%"


REM ---- PHASE 1: the TRAIN db ----
if exist "%DIR%\train5_%STAMP%.log" del /q "%DIR%\train5_%STAMP%.log"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_train_5k_h1_id5.toml > "%DIR%\train5_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TRAIN_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_13 %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/train_db_5k_h1_id5 100000 %WIDTH% >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TRAIN_BAD %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_14 %DATE% %TIME% >> "%LOG%"
  exit /b 14
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db_idfill.py F:/rwkv_lmdb/train_db_5k_h1_id5 400000000000 1 20 53 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TRAIN_BUGC_IN_ARTIFACT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_47 %DATE% %TIME% >> "%LOG%"
  exit /b 47
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db_cycles.py F:/rwkv_lmdb/train_db_5k_h1_id5 400000000000 1 12 53 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TRAIN_CYCLES_NOT_IN_ARTIFACT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_49 %DATE% %TIME% >> "%LOG%"
  exit /b 49
)
echo REBUILD5 TRAIN_OK %TIME% >> "%LOG%"

REM ---- PHASE 2: the EVAL db ----
if exist "%DIR%\test5_%STAMP%.log" del /q "%DIR%\test5_%STAMP%.log"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_test_5k_h2_id5.toml > "%DIR%\test5_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TEST_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_15 %DATE% %TIME% >> "%LOG%"
  exit /b 15
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/test_db_5k_id5 2000 %WIDTH% >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TEST_BAD %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_16 %DATE% %TIME% >> "%LOG%"
  exit /b 16
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db_idfill.py F:/rwkv_lmdb/test_db_5k_id5 250000000000 5001 20 53 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TEST_BUGC_IN_ARTIFACT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_47 %DATE% %TIME% >> "%LOG%"
  exit /b 47
)
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db_cycles.py F:/rwkv_lmdb/test_db_5k_id5 250000000000 5001 12 53 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo REBUILD5 TEST_CYCLES_NOT_IN_ARTIFACT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_49 %DATE% %TIME% >> "%LOG%"
  exit /b 49
)
echo REBUILD5 TEST_OK %TIME% >> "%LOG%"

REM ---- PHASE 3: gen 5 must score the SAME REVIEWS as gen 4 ----
REM Both are built against label_filter_db_id_e2s, and `size` IS the stored label_is_equalize
REM count, so identical per-user counts are REQUIRED. A difference is a build bug, not a dataset
REM property -- and it would break the size gate for the realcyc-vs-gen4base pair.
REM (Gen 4's phase 3 expected a DIFFERENCE, because gen 4 introduced the e2s filter. Same check,
REM meaning flipped by a config decision; revisited deliberately rather than cloned.)
REM Non-fatal by design: the dbs are built and verified by now, so a mismatch is information
REM for a human, not a reason to discard hours of work -- but it MUST be read before any run.
.venv\Scripts\python.exe scratchpad/features_rebuild/compare_equalize.py F:/rwkv_lmdb/test_db_5k_id4 250000000000 F:/rwkv_lmdb/test_db_5k_id5 250000000000 5001 20 53 >> "%LOG%" 2>&1
if %ERRORLEVEL%==0 (
  echo REBUILD5 EQUALIZE_MATCHES_GEN4 -- as REQUIRED, same label filter %TIME% >> "%LOG%"
) else (
  echo REBUILD5 EQUALIZE_DIFFERS_FROM_GEN4 -- BUILD BUG, do not run realcyc on this %DATE% %TIME% >> "%LOG%"
)

REM Terminal marker BEFORE endlocal: endlocal restores the pre-setlocal environment, so %LOG%
REM would expand to empty and the marker would be appended to "" instead.
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
