@echo off
REM muongates verdict, automatic: waits for muongates' runner to write its terminal marker, then runs the
REM pre-registered BOTH-modes gate against the control named in muongates/CONTROL.txt, then the P2
REM engagement probe gate_engagement.py (treated group vs groups whose optimizer did NOT change -- the
REM iter-67 lesson) at BelowNormal priority. Writes scratchpad/muongates/verdict.log. CRLF -- goto rule.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=4
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\muongates
set LOG=%DIR%\verdict.log
set MGLOG=%DIR%\muongates.log
set CTRL=realcyc
echo ===== muongates verdict waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitmg
if not exist "%MGLOG%" (
  ping -n 121 127.0.0.1 >nul
  goto waitmg
)
findstr /B /C:"DONE_EXIT_" "%MGLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitmg
)
findstr /C:"muongates EVAL_OK" "%MGLOG%" >nul 2>&1
if errorlevel 1 (
  echo muongates ended WITHOUT EVAL_OK -- no verdict %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_66 %DATE% %TIME% >> "%LOG%"
  exit /b 66
)
for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"control=" "%DIR%\CONTROL.txt"') do set CTRL=%%B
echo muongates EVAL_OK seen, control = %CTRL% %DATE% %TIME% >> "%LOG%"
echo ===== both-modes gate vs %CTRL% ===== >> "%LOG%"
.venv\Scripts\python.exe scratchpad/realcyc/realcyc_verdict.py muongates %CTRL% >> "%LOG%" 2>&1
echo gate rc %ERRORLEVEL% %TIME% >> "%LOG%"
if not exist "%DIR%\mg_d_10935.pth" (
  echo P2 SKIPPED: mg_d_10935.pth missing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_67 %DATE% %TIME% >> "%LOG%"
  exit /b 67
)
echo ===== P2 engagement: treated gates vs unchanged-optimizer control groups, at BelowNormal %DATE% %TIME% ===== >> "%LOG%"
start "" /belownormal /b /wait cmd /c ".venv\Scripts\python.exe -u scratchpad/muongates/gate_engagement.py >> "%LOG%" 2>&1"
echo probe rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
