@echo off
REM Re-measure the d=32 champion AUGMENTATION-OFF (fixed seed 1234) on the 100/100 workbench, TWICE,
REM to (a) establish the fp32 gate-reference logloss and (b) confirm run-to-run variance ~0 now that
REM augmentation is disabled. Recipe = route-R sc8k (8192-chunk db, MAX=66000, WS 6 epochs). Detached
REM (Esc/teardown-proof); monitor scratchpad/champ_off.log (poll for DONE_EXIT_).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\champ_off.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234

echo ================ CHAMP AUG-OFF START %DATE% %TIME% ================ > "%LOG%"

echo === WS run1 (aug-off) START %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/champ_off1_ws.toml >> "%LOG%" 2>&1
echo === WS run1 DONE %TIME% === >> "%LOG%"

echo === WS run2 (aug-off) START %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/champ_off2_ws.toml >> "%LOG%" 2>&1
echo === WS run2 DONE %TIME% === >> "%LOG%"

REM get_result SKIPS users already present in BOTH output files -> delete stale jsonls so it re-evals.
del /Q result\RWKV-champoff1.jsonl result\RWKV-P-champoff1.jsonl 2>nul
del /Q result\RWKV-champoff2.jsonl result\RWKV-P-champoff2.jsonl 2>nul

echo === EVAL run1 on 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/champ_off1_eval.toml >> "%LOG%" 2>&1
echo === EVAL run2 on 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/champ_off2_eval.toml >> "%LOG%" 2>&1

echo === COMPARISON === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/compare_champ_off.py >> "%LOG%" 2>&1

echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
