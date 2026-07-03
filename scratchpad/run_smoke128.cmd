@echo off
REM Pre-flight smoke for run_base5k_eval.cmd: SAME swap-restore + env-clearing machinery, pointed
REM at users 101-105 of the existing test_db with the known old128 checkpoint. Log: smoke128.log
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PY=.venv\Scripts\python.exe
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\smoke128.log
set PYTHONUNBUFFERED=1
set RWKV_N_HEADS=
set RWKV_HEAD_DIM=
set RWKV_QAT_LOWRANK_SCOPE=
set RWKV_QAT_PQ=
set RWKV_QAT_FUSED=
set RWKV_QAT_SHIFT_SCOPE=
set RWKV_NO_JIT=
set RWKV_COMPILE=
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0

echo ===== SMOKE128 START %DATE% %TIME% ===== > "%LOG%"
copy /y rwkv\architecture.py scratchpad\architecture_champion_backup.py >> "%LOG%" 2>&1
copy /y scratchpad\architecture_old_d128.py rwkv\architecture.py >> "%LOG%" 2>&1
del /q result\RWKV-smoke128.jsonl result\RWKV-P-smoke128.jsonl 2>nul

%PY% -u -m rwkv.get_result --config scratchpad/old128_smoke.toml >> "%LOG%" 2>&1
set EVAL_EXIT=%ERRORLEVEL%

copy /y scratchpad\architecture_champion_backup.py rwkv\architecture.py >> "%LOG%" 2>&1
echo [ARCH RESTORED] %DATE% %TIME% >> "%LOG%"
echo SMOKE128_EXIT_%EVAL_EXIT% %DATE% %TIME% >> "%LOG%"
