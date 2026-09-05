@echo off
REM sam verdict, automatic: waits for sam's runner to write its terminal marker, then runs
REM (1) the pre-registered gate script realcyc_verdict.py sam realcyc and (2) the P2 engagement probe
REM sam_probe.py on the SAM-decayed checkpoint at BelowNormal priority, because hord trains on the GPU by then
REM and a normal-priority CPU job costs a dispatch-bound step ~25 pct. Writes scratchpad/sam/verdict.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=6
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sam\verdict.log
set SAMLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sam\sam.log
set SAM_PROBE_CKPT=scratchpad/realcyc/sam_d_10935.pth
echo ===== sam verdict waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitsam
findstr /B /C:"DONE_EXIT_" "%SAMLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitsam
)
findstr /C:"sam EVAL_OK" "%SAMLOG%" >nul 2>&1
if errorlevel 1 (
  echo sam ended WITHOUT EVAL_OK -- no verdict %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_66 %DATE% %TIME% >> "%LOG%"
  exit /b 66
)
echo sam EVAL_OK seen, running the gate %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/realcyc/realcyc_verdict.py sam realcyc >> "%LOG%" 2>&1
echo gate rc %ERRORLEVEL% %TIME% >> "%LOG%"
if not exist "%SAM_PROBE_CKPT%" (
  echo P2 SKIPPED: %SAM_PROBE_CKPT% missing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_67 %DATE% %TIME% >> "%LOG%"
  exit /b 67
)
echo ===== P2 sharpness probe on %SAM_PROBE_CKPT% at BelowNormal %DATE% %TIME% ===== >> "%LOG%"
start "" /belownormal /b /wait cmd /c ".venv\Scripts\python.exe -u scratchpad/proposals_2026-09-04/sam_probe.py 12 >> "%LOG%" 2>&1"
echo probe rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
