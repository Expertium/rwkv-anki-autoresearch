@echo off
REM Decay run1's WS-final at 66000 (4 epochs) -> the re-baselined champion (rebase_champ/rebasec_680.pth).
REM Run AFTER run_rebaseline.cmd finishes (needs run1). The optim-name dance: WS saved rebase_optim_1020.pth
REM but the decay LOAD wants rebase_1020_optim.pth. Detached; ~14 min on a clean GPU.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\rebaseline_decay.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
echo === optim-copy + DECAY 66000 (4ep) from run1 WS-final START %DATE% %TIME% === > "%LOG%"
copy /Y scratchpad\rebase_run1\rebase_optim_1020.pth scratchpad\rebase_run1\rebase_1020_optim.pth >> "%LOG%" 2>&1
if exist scratchpad\rebase_champ rmdir /S /Q scratchpad\rebase_champ
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/rebase_66k_decay.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
