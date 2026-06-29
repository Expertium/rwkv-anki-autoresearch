@echo off
REM Eval OLD (RWKV_trained_on_5000_10000, d=128) and NEW (champion iter36) on users 1001-2000, then
REM compare. The OLD model needs the d=128 architecture -> swap architecture.py around its eval and
REM ALWAYS restore the champion afterward. Run AFTER scratchpad/build_eval1k.cmd finishes (DONE_EXIT_0).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eval1k.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
echo === NEW model (champion iter36) eval 1001-2000 START %DATE% %TIME% === > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config rwkv/get_result_new_1k.toml >> "%LOG%" 2>&1
echo === NEW DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
echo === OLD model (d=128) eval 1001-2000 (arch swap) START %TIME% === >> "%LOG%"
copy /Y scratchpad\architecture_old_d128.py rwkv\architecture.py >> "%LOG%" 2>&1
.venv\Scripts\python.exe -u -m rwkv.get_result --config rwkv/get_result_old_1k.toml >> "%LOG%" 2>&1
echo === OLD DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
copy /Y scratchpad\architecture_champion_backup.py rwkv\architecture.py >> "%LOG%" 2>&1
echo (champion arch restored) >> "%LOG%"
echo === COMPARISON === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/compare_eval1k.py >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
