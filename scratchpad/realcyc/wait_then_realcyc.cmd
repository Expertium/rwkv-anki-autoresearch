@echo off
REM =========================================================================================
REM Launch the realcyc run when BOTH of its preconditions hold:
REM   1. the GPU is free -- the feature-ablation chain (queued behind gen4base) has reported;
REM   2. generation 5 exists -- rebuild5.log carries DONE_EXIT_0, SUCCESS specifically. A failed
REM      rebuild must not launch a 10 h run against a half-built db; and a db that exists but
REM      failed its cycle/idfill checks would produce a plausible number for the wrong data.
REM
REM Two markers, two files, both anchored. The ablation's marker means "gen4base finished AND the
REM ablation finished", because the ablation waiter itself waits on gen4base -- so the GPU is
REM genuinely idle when this fires, not merely between two chained jobs.
REM
REM Anchored findstr -- the unanchored form matches this file's own prose and fires instantly.
REM No angle brackets or arrows in REM lines.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\wait_realcyc.log
set GPULOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_ablate\wait_ablate.log
set DBLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\rebuild5.log

echo ===== realcyc waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitboth
findstr /B /C:"DONE_EXIT_" "%GPULOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitboth
)
findstr /B /C:"DONE_EXIT_0" "%DBLOG%" >nul 2>&1
if errorlevel 1 (
  findstr /B /C:"DONE_EXIT_" "%DBLOG%" >nul 2>&1
  if not errorlevel 1 (
    echo gen 5 rebuild FAILED -- refusing to launch realcyc %DATE% %TIME% >> "%LOG%"
    echo DONE_EXIT_62 %DATE% %TIME% >> "%LOG%"
    exit /b 62
  )
  ping -n 121 127.0.0.1 >nul
  goto waitboth
)
echo GPU free and gen 5 built %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\realcyc\run_realcyc.cmd
echo run_realcyc returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
