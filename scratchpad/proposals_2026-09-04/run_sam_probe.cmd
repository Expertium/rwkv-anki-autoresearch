@echo off
REM SAM sharpness screen on realcyc (ranked-queue rank 6), 12 train-range chunks, CPU only (~15 min).
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=6
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\proposals_2026-09-04\sam_probe.log
echo ===== SAM PROBE START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad/proposals_2026-09-04/sam_probe.py 12 >> "%LOG%" 2>&1
echo probe rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
