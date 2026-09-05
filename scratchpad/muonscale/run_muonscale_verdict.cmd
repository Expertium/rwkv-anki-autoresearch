@echo off
REM muonscale verdict, automatic: waits for muonscale's runner to write its terminal marker, then runs the
REM pre-registered BOTH-modes gate (realcyc_verdict.py muonscale CONTROL) against the control named in
REM muonscale/CONTROL.txt (control=hord or control=realcyc, written by auto_control.py at launch), then the P2
REM engagement probe scale_probe.py on ms_ws_50 vs ms_d_10935 (median anisotropy ratio must fall below 0.55) at
REM BelowNormal priority. Writes scratchpad/muonscale/verdict.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=4
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\muonscale
set LOG=%DIR%\verdict.log
set MSLOG=%DIR%\muonscale.log
set INIT_CKPT=scratchpad/muonscale/ms_ws_50.pth
set FINAL_CKPT=scratchpad/muonscale/ms_d_10935.pth
set CTRL=realcyc
echo ===== muonscale verdict waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitms
findstr /B /C:"DONE_EXIT_" "%MSLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitms
)
findstr /C:"muonscale EVAL_OK" "%MSLOG%" >nul 2>&1
if errorlevel 1 (
  echo muonscale ended WITHOUT EVAL_OK -- no verdict %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_66 %DATE% %TIME% >> "%LOG%"
  exit /b 66
)
for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"control=" "%DIR%\CONTROL.txt"') do set CTRL=%%B
echo muonscale EVAL_OK seen, control = %CTRL% %DATE% %TIME% >> "%LOG%"
echo ===== both-modes gate vs %CTRL% ===== >> "%LOG%"
.venv\Scripts\python.exe scratchpad/realcyc/realcyc_verdict.py muonscale %CTRL% >> "%LOG%" 2>&1
echo gate rc %ERRORLEVEL% %TIME% >> "%LOG%"
if not exist "%FINAL_CKPT%" (
  echo P2 SKIPPED: %FINAL_CKPT% missing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_67 %DATE% %TIME% >> "%LOG%"
  exit /b 67
)
if not exist "%INIT_CKPT%" (
  echo P2 SKIPPED: %INIT_CKPT% missing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_68 %DATE% %TIME% >> "%LOG%"
  exit /b 68
)
echo ===== P2 update-anisotropy probe %INIT_CKPT% to %FINAL_CKPT% at BelowNormal -- scale tensors' median ratio must fall below 0.55, realcyc 0.653 %DATE% %TIME% ===== >> "%LOG%"
start "" /belownormal /b /wait cmd /c ".venv\Scripts\python.exe -u scratchpad/proposals_2026-09-04/scale_probe.py %INIT_CKPT% %FINAL_CKPT% >> "%LOG%" 2>&1"
echo probe rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
