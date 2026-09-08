@echo off
REM Dump per-review FSRS predictions for the out-of-family decorrelation test.
REM Three arms, all CPU (config.py sends FSRS to cpu unconditionally), so nothing competes
REM with the ws10 GPU run:
REM   fsrs7    FSRS-7, parameters fitted PER USER  == the leaderboard configuration
REM   fsrs6    FSRS-6, parameters fitted PER USER  == the WITHIN-family calibration:
REM            what residual correlation does a mere version bump produce?
REM   fsrs7def FSRS-7 with DEFAULT parameters      == the personalization control, so a
REM            decorrelation can be attributed to inductive bias rather than to per-user fitting.
REM Code is read from Andrew's srs-benchmark clone and nothing is written there: the driver
REM chdir's into its own output dir before importing, so relative output paths land here.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PY=C:\Users\Andrew\srs-benchmark\.venv\Scripts\python.exe
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fsrs7
set LOG=%DIR%\fsrs7.log
set MPLBACKEND=Agg
echo ===== FSRS DUMP START %DATE% %TIME% ===== > "%LOG%"

echo --- ARM fsrs7 %TIME% >> "%LOG%"
"%PY%" -u scratchpad/fsrs7/dump_fsrs.py fsrs7 -- --algo FSRS-7 --short --secs >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_11 %DATE% %TIME% >> "%LOG%"
  exit /b 11
)
echo --- ARM fsrs6 %TIME% >> "%LOG%"
"%PY%" -u scratchpad/fsrs7/dump_fsrs.py fsrs6 -- --algo FSRS-6 --short --secs >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_12 %DATE% %TIME% >> "%LOG%"
  exit /b 12
)
echo --- ARM fsrs7def %TIME% >> "%LOG%"
"%PY%" -u scratchpad/fsrs7/dump_fsrs.py fsrs7def -- --algo FSRS-7 --short --secs --default >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_13 %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
echo FSRS DUMP OK %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
