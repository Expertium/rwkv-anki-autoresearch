@echo off
REM ============================================================================
REM anki-revlogs-10k-id build pipeline (Andrew 2026-07-15): download the raw HF
REM dataset (revlogs.7z, 8,459,427,959 bytes), extract, and build parquets with
REM REAL IDS + the review-time correction (review_time = id - taken_millis; see
REM build_parquet_id.py docstring). Resumable at every step. The CPU-heavy build
REM step WAITS for iter18's pipeline to finish (DONE_EXIT in its log) so the
REM 6-proc pool never contends with the running eval.
REM Staging: C:\Users\Andrew\anki-revlogs-10k-id-raw  Output: C:\Users\Andrew\anki-revlogs-10k-id
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dataset_id\build_id.log
set PYTHONUNBUFFERED=1
set RAW=C:\Users\Andrew\anki-revlogs-10k-id-raw
set OUT=C:\Users\Andrew\anki-revlogs-10k-id

echo ===== BUILD_10K_ID START %DATE% %TIME% ===== > "%LOG%"
if not exist "%RAW%" mkdir "%RAW%"

echo === STEP 1: download revlogs.7z (resumable) %TIME% === >> "%LOG%"
curl -L -C - --retry 10 --retry-delay 15 -o "%RAW%\revlogs.7z" "https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k-raw/resolve/main/revlogs.7z" >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
for %%A in ("%RAW%\revlogs.7z") do set SZ=%%~zA
echo downloaded size: %SZ% (expect 8459427959) >> "%LOG%"
if not "%SZ%"=="8459427959" (
  echo DONE_EXIT_SIZEMISMATCH %DATE% %TIME% >> "%LOG%"
  exit /b 3
)

echo === STEP 2: extract %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad\dataset_id\extract_7z.py "%RAW%\revlogs.7z" "%RAW%\revlogs" >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EXTRACTFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 4
)

echo === STEP 3: wait for iter18 pipeline to release the CPU %TIME% === >> "%LOG%"
:waitloop
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter18_nodur\iter18_nodur.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 60 /nobreak >nul
  goto waitloop
)
echo iter18 done -- starting build %TIME% >> "%LOG%"

echo === STEP 4: build parquets (real ids + corrected review_time, 6 procs) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad\dataset_id\build_parquet_id.py "%RAW%\revlogs" "%OUT%" 6 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_BUILDFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
