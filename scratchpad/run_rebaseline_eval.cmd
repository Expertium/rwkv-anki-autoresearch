@echo off
REM Eval pipeline for the re-baseline, run AFTER run_rebaseline.cmd finishes (clean GPU -- no contention).
REM Variance (run1/run2 on 101-200) + 3-way old/5%-champ/re-baseline on 1001-2000 + compare. Deletes stale
REM result jsonls first (get_result SKIPS users already present in BOTH output files -> stale = no rewrite).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\rebaseline_eval.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
del /Q result\RWKV-rb1-100.jsonl result\RWKV-P-rb1-100.jsonl result\RWKV-rb2-100.jsonl result\RWKV-P-rb2-100.jsonl 2>nul
del /Q result\RWKV-rb-1k.jsonl result\RWKV-P-rb-1k.jsonl result\RWKV-iter45-1k.jsonl result\RWKV-P-iter45-1k.jsonl 2>nul
del /Q result\RWKV-old-1k.jsonl result\RWKV-P-old-1k.jsonl 2>nul
echo === variance evals (101-200) %TIME% === > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config rwkv/get_result_rb1_100.toml >> "%LOG%" 2>&1
.venv\Scripts\python.exe -u -m rwkv.get_result --config rwkv/get_result_rb2_100.toml >> "%LOG%" 2>&1
echo === 1001-2000: re-baselined champion (run1) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config rwkv/get_result_rb_1k.toml >> "%LOG%" 2>&1
echo === 1001-2000: current champion iter45 fp32 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config rwkv/get_result_iter45_1k.toml >> "%LOG%" 2>&1
echo === 1001-2000: OLD model (d=128 arch swap) %TIME% === >> "%LOG%"
copy /Y scratchpad\architecture_old_d128.py rwkv\architecture.py >> "%LOG%" 2>&1
.venv\Scripts\python.exe -u -m rwkv.get_result --config rwkv/get_result_old_1k.toml >> "%LOG%" 2>&1
copy /Y scratchpad\architecture_champion_backup.py rwkv\architecture.py >> "%LOG%" 2>&1
echo === COMPARISON === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/compare_rebaseline.py >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
