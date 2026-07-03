@echo off
REM Task #18 irreducible-entropy: RAW evals of BOTH disjoint-trained d=128 models on users 1-100.
REM Swaps in the OLD architecture, runs A then B, ALWAYS restores. Log: scratchpad/floor_est.log
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PY=.venv\Scripts\python.exe
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\floor_est.log
set PYTHONUNBUFFERED=1
set RWKV_N_HEADS=
set RWKV_HEAD_DIM=
set RWKV_QAT_LOWRANK_SCOPE=
set RWKV_QAT_PQ=
set RWKV_QAT_FUSED=
set RWKV_QAT_SHIFT_SCOPE=
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0

echo ===== FLOOR EST START %DATE% %TIME% ===== >> "%LOG%"
copy /y rwkv\architecture.py scratchpad\architecture_champion_backup.py >> "%LOG%" 2>&1
copy /y scratchpad\architecture_old_d128.py rwkv\architecture.py >> "%LOG%" 2>&1
del /q result\RWKV-floorA.jsonl result\RWKV-P-floorA.jsonl raw\RWKV-floorA.jsonl raw\RWKV-P-floorA.jsonl 2>nul
del /q result\RWKV-floorB.jsonl result\RWKV-P-floorB.jsonl raw\RWKV-floorB.jsonl raw\RWKV-P-floorB.jsonl 2>nul

%PY% -u -m rwkv.get_result --config scratchpad/floorA.toml >> "%LOG%" 2>&1
set EXIT_A=%ERRORLEVEL%
echo [FLOOR_A_EXIT_%EXIT_A%] %DATE% %TIME% >> "%LOG%"

%PY% -u -m rwkv.get_result --config scratchpad/floorB.toml >> "%LOG%" 2>&1
set EXIT_B=%ERRORLEVEL%
echo [FLOOR_B_EXIT_%EXIT_B%] %DATE% %TIME% >> "%LOG%"

copy /y scratchpad\architecture_champion_backup.py rwkv\architecture.py >> "%LOG%" 2>&1
echo [ARCH RESTORED] %DATE% %TIME% >> "%LOG%"
echo FLOOR_EVALS_DONE_A%EXIT_A%_B%EXIT_B% %DATE% %TIME% >> "%LOG%"
