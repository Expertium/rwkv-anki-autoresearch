@echo off
REM ordcut P3 (second half): the shared screen instrument on the ordcut DECAYED checkpoint, same 10
REM train users as the realcyc baseline. PREREG: AUC(Good vs Hard) on the candidate's own logit R must
REM RISE above realcyc's 0.737. CPU only, detached, BelowNormal beside the GPU eval. Log: screen_ordcut.log
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=4
set REUSE=0
set SCREEN_CKPT=scratchpad/ordcut/oc_d_10935.pth
set SCREEN_OUT=scratchpad/ordcut/screen_records_ordcut.npz
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ordcut\screen_ordcut.log
echo ===== ORDCUT SCREEN START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad/proposals_2026-09-04/screen_pass.py 107 136 156 178 203 1207 2207 3207 4207 2707 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo screen FAILED rc %ERRORLEVEL% %TIME% >> "%LOG%"
  echo DONE_EXIT_1 %DATE% %TIME% >> "%LOG%"
  exit /b 1
)
echo screen OK %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
