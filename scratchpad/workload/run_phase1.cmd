@echo off
REM =========================================================================================
REM Workload-efficiency replay, phase 1: 24 users from the VAL half, 416k reviews.
REM Two workers, one thread each, so this job costs 2 CPU threads and no GPU at all.
REM
REM Detached via scratchpad/detach.ps1 so Esc cannot tree-kill it.
REM Terminal marker is written BEFORE endlocal: endlocal restores the pre-setlocal
REM environment, so LOG would expand to empty and the marker would vanish.
REM No angle brackets, arrows or pipes in REM lines: cmd.exe parses redirection before REM.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\workload\phase1.log
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1

echo ===== WORKLOAD PHASE 1 START %DATE% %TIME% ===== >> "%LOG%"

if not exist "scratchpad\workload\users_phase1.json" (
  echo PHASE1 USERS_JSON_MISSING %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_31 %DATE% %TIME% >> "%LOG%"
  exit /b 31
)

.venv\Scripts\python.exe -u scratchpad\workload\run_pipeline.py scratchpad\workload\users_phase1.json 2 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo PHASE1 PIPELINE_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_25 %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo PHASE1 PIPELINE_OK %TIME% >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
