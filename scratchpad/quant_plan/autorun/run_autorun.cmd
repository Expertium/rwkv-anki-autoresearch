@echo off
rem Hourly quant supervisor wrapper. Called by the Windows Task Scheduler task RWKV-Quant-Autorun.
rem It takes a single-instance lock, runs the goose recipe headless, logs stdout and the exit code,
rem then releases the lock. See QUEUE.md for what the run may do.
setlocal
set REPO=C:\Users\Andrew\rwkv-anki-autoresearch
set AD=%REPO%\scratchpad\quant_plan\autorun
set GOOSE=C:\Users\Andrew\AppData\Local\Programs\Goose\resources\bin\goose.exe
set PY=%REPO%\.venv\Scripts\python.exe
set RUNLOG=%AD%\autorun_runs.log

cd /d "%REPO%"
echo. >> "%RUNLOG%"
echo ==== %DATE% %TIME% autorun wrapper start ==== >> "%RUNLOG%"

"%PY%" "%AD%\autorun_lock.py" check >> "%RUNLOG%" 2>&1
if errorlevel 1 (
  echo wrapper: lock is held, nothing to do >> "%RUNLOG%"
  exit /b 3
)
"%PY%" "%AD%\autorun_lock.py" take >> "%RUNLOG%" 2>&1

"%GOOSE%" run --recipe "%AD%\quant_hourly.yaml" --no-session --max-turns 80 >> "%RUNLOG%" 2>&1
set RC=%errorlevel%

"%PY%" "%AD%\autorun_lock.py" release >> "%RUNLOG%" 2>&1
echo ==== %DATE% %TIME% autorun wrapper end rc=%RC% ==== >> "%RUNLOG%"
exit /b %RC%
