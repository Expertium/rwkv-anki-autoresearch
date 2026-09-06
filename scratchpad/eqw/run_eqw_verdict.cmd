@echo off
REM eqw verdict, automatic: waits for eqw's runner to write its terminal marker, then runs the pre-registered
REM BOTH-modes gate (realcyc_verdict.py eqw CONTROL) against the control named in eqw/CONTROL.txt, then the P2
REM engagement screen: a deploy-RNN pass on the eqw checkpoint over the same 10 train users (screen_pass.py with
REM SCREEN_CKPT/SCREEN_OUT, BelowNormal, ~40 min) and p2_history_split.py against realcyc's records.
REM Writes scratchpad/eqw/verdict.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=4
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eqw
set LOG=%DIR%\verdict.log
set EQLOG=%DIR%\eqw.log
set CKPT=scratchpad/eqw/eq_d_10935.pth
set CTRL=realcyc
echo ===== eqw verdict waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waiteq
if not exist "%EQLOG%" (
  ping -n 121 127.0.0.1 >nul
  goto waiteq
)
findstr /B /C:"DONE_EXIT_" "%EQLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waiteq
)
findstr /C:"eqw EVAL_OK" "%EQLOG%" >nul 2>&1
if errorlevel 1 (
  echo eqw ended WITHOUT EVAL_OK -- no verdict %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_66 %DATE% %TIME% >> "%LOG%"
  exit /b 66
)
for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"control=" "%DIR%\CONTROL.txt"') do set CTRL=%%B
echo eqw EVAL_OK seen, control = %CTRL% %DATE% %TIME% >> "%LOG%"
echo ===== both-modes gate vs %CTRL% ===== >> "%LOG%"
.venv\Scripts\python.exe scratchpad/realcyc/realcyc_verdict.py eqw %CTRL% >> "%LOG%" 2>&1
echo gate rc %ERRORLEVEL% %TIME% >> "%LOG%"
if not exist "%CKPT%" (
  echo P2 SKIPPED: %CKPT% missing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_67 %DATE% %TIME% >> "%LOG%"
  exit /b 67
)
echo ===== P2 deploy-RNN pass on %CKPT% at BelowNormal, then the history split vs realcyc %DATE% %TIME% ===== >> "%LOG%"
set SCREEN_CKPT=%CKPT%
set SCREEN_OUT=scratchpad/eqw/screen_records_eqw.npz
set REUSE=0
start "" /belownormal /b /wait cmd /c ".venv\Scripts\python.exe -u scratchpad/proposals_2026-09-04/screen_pass.py >> "%LOG%" 2>&1"
echo screen rc %ERRORLEVEL% %TIME% >> "%LOG%"
.venv\Scripts\python.exe scratchpad/eqw/p2_history_split.py scratchpad/eqw/screen_records_eqw.npz scratchpad/proposals_2026-09-04/screen_records.npz >> "%LOG%" 2>&1
echo p2 rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
