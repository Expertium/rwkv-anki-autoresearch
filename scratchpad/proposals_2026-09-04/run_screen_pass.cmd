@echo off
REM The shared CPU screen for ranked-queue items 1, 2 and 10 (2026-09-04): one deploy-RNN pass on
REM realcyc over 10 train-range users of the -id set. Detached so an Esc cannot kill it. ~40 min.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=4
set REUSE=0
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\proposals_2026-09-04\screen_pass.log
echo ===== SCREEN PASS START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad/proposals_2026-09-04/screen_pass.py 107 136 156 178 203 1207 2207 3207 4207 2707 >> "%LOG%" 2>&1
echo screen rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
