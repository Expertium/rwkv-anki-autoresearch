@echo off
REM hord verdict, automatic: waits for hord's runner to write its terminal marker, then runs the pre-registered
REM CURVE-SIDE gate vs the control named in hord/CONTROL.txt (control=sam or control=realcyc, written by
REM auto_control.py at launch): size check, paired_pvalue --curve-side, realcyc_verdict.py for the means and
REM per-mode deltas (its both-modes gate line is INFORMATIONAL for a curve-side lever), then the P3 engagement
REM probe button_probe.py on hord's decayed checkpoint at BelowNormal priority, because muonscale trains on the
REM GPU by then. Writes scratchpad/hord/verdict.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=4
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hord
set LOG=%DIR%\verdict.log
set HORDLOG=%DIR%\hord.log
set BUTTON_PROBE_CKPT=scratchpad/hord/hd_d_10935.pth
set CTRL=realcyc
echo ===== hord verdict waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waithord
findstr /B /C:"DONE_EXIT_" "%HORDLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waithord
)
findstr /C:"hord EVAL_OK" "%HORDLOG%" >nul 2>&1
if errorlevel 1 (
  echo hord ended WITHOUT EVAL_OK -- no verdict %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_66 %DATE% %TIME% >> "%LOG%"
  exit /b 66
)
for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"control=" "%DIR%\CONTROL.txt"') do set CTRL=%%B
echo hord EVAL_OK seen, control = %CTRL% %DATE% %TIME% >> "%LOG%"
echo ===== size gate ===== >> "%LOG%"
.venv\Scripts\python.exe optimization/size_baseline.py check id_e2s result/RWKV-hord.jsonl >> "%LOG%" 2>&1
echo size rc %ERRORLEVEL% >> "%LOG%"
echo ===== curve-side gate vs %CTRL% ===== >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-hord.jsonl --cand-imm result/RWKV-P-hord.jsonl --champ-ahead result/RWKV-%CTRL%.jsonl --champ-imm result/RWKV-P-%CTRL%.jsonl --intersect --curve-side >> "%LOG%" 2>&1
echo curve-side rc %ERRORLEVEL% >> "%LOG%"
echo ===== means and deltas vs %CTRL% -- the both-modes gate line below is INFORMATIONAL ===== >> "%LOG%"
.venv\Scripts\python.exe scratchpad/realcyc/realcyc_verdict.py hord %CTRL% >> "%LOG%" 2>&1
echo verdict-script rc %ERRORLEVEL% >> "%LOG%"
if not exist "%BUTTON_PROBE_CKPT%" (
  echo P3 SKIPPED: %BUTTON_PROBE_CKPT% missing %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_67 %DATE% %TIME% >> "%LOG%"
  exit /b 67
)
echo ===== P3 button-order probe on %BUTTON_PROBE_CKPT% at BelowNormal -- crossing rates must fall at least 50 pct vs realcyc 32.5/32.1/35.6/48.8 %DATE% %TIME% ===== >> "%LOG%"
start "" /belownormal /b /wait cmd /c ".venv\Scripts\python.exe -u scratchpad/proposals_2026-09-04/button_probe.py >> "%LOG%" 2>&1"
echo probe rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
