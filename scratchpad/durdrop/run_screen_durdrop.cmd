@echo off
REM durdrop P3 (engagement): the shared screen instrument on the durdrop DECAYED checkpoint, same 10
REM train users as the realcyc baseline (+0.001388 duration-zeroing cost). PREREG: must fall under
REM +0.0007. CPU only, detached, run at BelowNormal beside the GPU eval. Log: screen_durdrop.log
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=4
set REUSE=0
set SCREEN_CKPT=scratchpad/durdrop/dd_d_10935.pth
set SCREEN_OUT=scratchpad/durdrop/screen_records_durdrop.npz
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\durdrop\screen_durdrop.log
echo ===== DURDROP SCREEN START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad/proposals_2026-09-04/screen_pass.py 107 136 156 178 203 1207 2207 3207 4207 2707 >> "%LOG%" 2>&1
echo screen rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
