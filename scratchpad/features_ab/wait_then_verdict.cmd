@echo off
REM ===========================================================================================
REM Run the INTERVAL VERDICT the moment the fixc arm lands.
REM
REM interval_verdict.py is CPU-only and takes seconds, so this runs alongside featB (which the
REM other waiter starts on the same trigger) without contending for anything that matters.
REM
REM The analysis was written BEFORE either arm reported and is not edited here -- that is the
REM point of having pre-registered it. It tests the three predictions in
REM scratchpad/features_ab/e2s/PREREG.md and prints the verdict, including the one I expect to be
REM wrong (that imm degrades more than ahead).
REM
REM Anchored findstr -- the unanchored form matches this file's own prose and fires instantly.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\interval_verdict.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fixc_arm\fixc_arm.log

echo ===== verdict waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo fixc arm finished %DATE% %TIME% >> "%LOG%"

REM Only meaningful if fixc SUCCEEDED -- a failed arm leaves no results to pair against.
findstr /B /C:"DONE_EXIT_0" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  echo fixc did NOT succeed -- no verdict to compute >> "%LOG%"
  echo DONE_EXIT_61 %DATE% %TIME% >> "%LOG%"
  exit /b 61
)

echo. >> "%LOG%"
.venv\Scripts\python.exe scratchpad/features_ab/interval_verdict.py >> "%LOG%" 2>&1
echo. >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-e2sc.jsonl --cand-imm result/RWKV-P-e2sc.jsonl --champ-ahead result/RWKV-fixc.jsonl --champ-imm result/RWKV-P-fixc.jsonl --intersect >> "%LOG%" 2>&1

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
