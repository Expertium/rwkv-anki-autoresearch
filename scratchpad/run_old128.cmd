@echo off
REM Baseline-to-beat: old d=128 arch trained from scratch on 1-100, eval 101-200. Backs up the CURRENT
REM (champion) architecture.py, swaps in the d=128 arch, trains+evals, then RESTORES the champion arch
REM (always, even if the run errors). Launch ONLY after the A/B pipeline finishes (GPU + arch.py free).
REM Detached/Esc-proof; monitor scratchpad/old128.log for DONE_EXIT.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\old128.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1

echo ================ OLD d=128 BASELINE START %DATE% %TIME% ================ > "%LOG%"
echo === backup champion arch + swap in d=128 %TIME% === >> "%LOG%"
copy /Y rwkv\architecture.py scratchpad\arch_champion_now.py >> "%LOG%" 2>&1
copy /Y scratchpad\architecture_old_d128.py rwkv\architecture.py >> "%LOG%" 2>&1
echo === train d=128 on 1-100 + eval 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/run_old128.py >> "%LOG%" 2>&1
echo === restore champion arch %TIME% === >> "%LOG%"
copy /Y scratchpad\arch_champion_now.py rwkv\architecture.py >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
