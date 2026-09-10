@echo off
REM Launch ENDGAME ARM 2 once its DRY RUN has proven the vcvars fix end to end.
REM
REM WHY A DRY RUN GATES IT. The 04:00 launch of arm 2 was HOLLOW -- 255 of 288 batches raised
REM "InductorError: Compiler: cl is not found" and train_rwkv's per-batch except swallowed every
REM one, so it stepped at 3.7 steps/min on ~11% of the data while the banners and the qat-assert
REM PASS all looked correct. RWKV_QAT_COMPILE=1 plus QAT is a combination that had never run here.
REM
REM THE GATE IS THREE CONDITIONS, not one. A terminal marker alone is satisfied by a failure, and
REM DONE_EXIT_0 alone is satisfied by a HOLLOW run that skipped most of its batches and still
REM finished. So this also requires ZERO tracebacks in the dry decay log -- which is the specific
REM defect being tested for -- and the dry eval's 20 result rows.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\wait_dry.log
set DRYLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qatdry\w10qatdry.log
set DRYDIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qatdry
echo ===== arm-2 waiter armed on the dry run %DATE% %TIME% ===== >> "%LOG%"

:waitdry
if not exist "%DRYLOG%" (
  echo dry log missing -- refusing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_80 %DATE% %TIME% >> "%LOG%"
  exit /b 80
)
findstr /B /C:"DONE_EXIT_" "%DRYLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 61 127.0.0.1 >nul
  goto waitdry
)

REM ---- gate 1: the dry run must have SUCCEEDED ----
findstr /B /C:"DONE_EXIT_0 " "%DRYLOG%" >nul 2>&1
if errorlevel 1 (
  echo dry run ended WITHOUT success -- not launching arm 2 %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_81 %DATE% %TIME% >> "%LOG%"
  exit /b 81
)

REM ---- gate 2: ZERO tracebacks. This is the defect under test; a hollow run exits 0. ----
findstr /C:"Traceback" "%DRYDIR%\decay_*.log" >nul 2>&1
if not errorlevel 1 (
  echo dry decay contains a Traceback -- the hollow-run defect is NOT fixed %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_82 %DATE% %TIME% >> "%LOG%"
  exit /b 82
)

REM ---- gate 3: the dry eval actually produced its 20 rows ----
if not exist "C:\Users\Andrew\rwkv-anki-autoresearch\result\RWKV-w10qatdry.jsonl" (
  echo dry eval produced no result file %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_83 %DATE% %TIME% >> "%LOG%"
  exit /b 83
)

echo dry run clean -- launching arm 2 %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat\run_w10qat.cmd
echo run_w10qat returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
