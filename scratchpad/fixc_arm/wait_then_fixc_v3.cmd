@echo off
REM ===========================================================================================
REM After the e2s re-base AND the upward MAX sweep: build the fixc control db ONTO THE SSD, then run the fixc arm.
REM
REM ---- THE OPTIMIZATION, AND WHY IT NEEDS NO PATH EDITS ----
REM The fixc tomls, the arm's dump/ws tomls, the decay-setup argument and the runner's own
REM findstr guards all name F:/rwkv_lmdb/train_db_5k_h1_fixc. Editing that string in five places
REM is the "clone a runner, update the lever but not everything that depends on it" failure this
REM project keeps recording. So instead the junction is created BEFORE the rebuild writes
REM anything: data_processing opens the F: path, the filesystem redirects, and the bytes land on
REM C:. Every toml and every guard keeps resolving, unchanged.
REM
REM WHY IT IS WORTH IT: reading a db from the USB drive costs 2.2x per step (measured today --
REM the C:-hosted teacher dump ran 1.40 steps/s, the same dump on F: 0.63, GPU utilisation 8%).
REM The train db is read by the dump, WS and decay phases, so this saves roughly 10 h on this arm.
REM The TEST db stays on F: -- it is touched only by the eval, and C: does not have room for both.
REM The fixc KD dump also stays on F: for the same reason; dump WRITES were measured NOT to be the
REM bottleneck (F: does 47.7 of those files per second, and the dump needs 1.4).
REM
REM ---- THE THRESHOLD IS 150 GB AND IT IS DELIBERATELY NOT MET ----
REM C: ends near 135 GB after the two e2s dbs, and the fixc train db is 103 GB, so a junction
REM would leave about 32 GB. The pagefile is ALREADY allocated at 73.6 GB on C: with a 34.3 GB
REM peak and is auto-managed, so it can still grow -- 32 GB is thinner than it looks. The gate is
REM therefore set ABOVE what is available on purpose: the fast path arms itself only if space
REM appears (e.g. a db is retired), and otherwise the arm builds on F: and simply runs longer.
REM
REM ---- GRACEFUL DEGRADATION, NOT A HARD FAILURE ----
REM If C: does not have room, the junction is skipped and the rebuild simply writes to F: as
REM originally planned: slower, correct, and it still runs unattended. A speed optimization must
REM never be able to stop the experiment.
REM
REM The rebuild is RESUMABLE (data_processing skips users already present), but note the partial
REM test db it already wrote lives on F: -- only the TRAIN db gets the junction, so nothing that
REM already exists is orphaned.
REM
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fixc_arm\wait_fixc_v3.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch\sweep_up.log
set FIXCTRAIN=F:\rwkv_lmdb\train_db_5k_h1_fixc
set FIXCTARGET=C:\rwkv_lmdb\train_db_5k_h1_fixc

echo ===== fixc chain v2 armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo sweep finished, GPU free %DATE% %TIME% >> "%LOG%"

REM The re-base must have SUCCEEDED. Running the control without its treatment produces a number
REM with nothing to compare it to.
findstr /B /C:"DONE_EXIT_0" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\e2s_rebase\e2s_rebase.log >nul 2>&1
if errorlevel 1 (
  echo REBASE_FAILED -- not running the control without its treatment >> "%LOG%"
  echo DONE_EXIT_53 %DATE% %TIME% >> "%LOG%"
  exit /b 53
)

REM ---- put the fixc TRAIN db on the SSD, if there is room ----
if exist "%FIXCTRAIN%" (
  echo fixc train path already exists -- leaving it alone >> "%LOG%"
  goto build
)
.venv\Scripts\python.exe scratchpad/workload/free_gb_gate.py C 150 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo C: too small for the fixc train db -- building on F: as originally planned >> "%LOG%"
  goto build
)
if not exist "%FIXCTARGET%" mkdir "%FIXCTARGET%"
mklink /J "%FIXCTRAIN%" "%FIXCTARGET%" >> "%LOG%" 2>&1
if not exist "%FIXCTRAIN%" (
  echo junction failed -- building on F: as originally planned >> "%LOG%"
) else (
  echo fixc train db will be written THROUGH a junction onto the SSD >> "%LOG%"
)

:build
echo --- resuming the fixc control rebuild %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\run_fixc_rebuild.cmd
findstr /B /C:"DONE_EXIT_0" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\fixc_rebuild.log >nul 2>&1
if errorlevel 1 (
  echo FIXC_REBUILD_FAILED >> "%LOG%"
  echo DONE_EXIT_54 %DATE% %TIME% >> "%LOG%"
  exit /b 54
)
echo fixc dbs built %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fixc_arm\run_fixc_arm.cmd
echo fixc arm returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
