@echo off
REM =========================================================================================
REM Run the featB verdict the moment featB lands.
REM
REM featB_verdict.py is CPU-only and takes seconds, so it does not contend with the gen-4
REM rebuild that the other waiter starts on the same trigger.
REM
REM The analysis was written BEFORE featB reported and is NOT edited here -- that is the whole
REM point of having pre-registered it. It tests the three predictions in featB/PREREG.md and
REM prints the verdict, including P2, whose direction is genuinely uncertain because the
REM interval penalty concentrates in the same users the features should help most.
REM
REM Anchored findstr -- the unanchored form matches this file's own prose and fires instantly.
REM No angle brackets or arrows in REM lines: cmd parses redirection before it honours REM.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featB_verdict.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featB\featB.log

echo ===== featB verdict waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo featB reported %DATE% %TIME% >> "%LOG%"

REM Only meaningful if featB SUCCEEDED -- a failed arm leaves no results to pair against.
findstr /B /C:"DONE_EXIT_0" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  echo featB did NOT succeed -- no verdict to compute >> "%LOG%"
  echo DONE_EXIT_61 %DATE% %TIME% >> "%LOG%"
  exit /b 61
)

echo. >> "%LOG%"
.venv\Scripts\python.exe scratchpad/features_ab/featB_verdict.py >> "%LOG%" 2>&1
echo. >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-featB.jsonl --cand-imm result/RWKV-P-featB.jsonl --champ-ahead result/RWKV-featA2.jsonl --champ-imm result/RWKV-P-featA2.jsonl --intersect >> "%LOG%" 2>&1

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
