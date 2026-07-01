@echo off
REM "Varied data, few epochs": train 1 epoch WS on 1500 users (1000-2499), eval 101-200, score vs champion
REM (0.314807/0.280200). Compute ~= 15 epochs/100u. Runs AFTER build_1500 finishes. Detached. Monitor
REM scratchpad/exp_1500.log (DONE_EXIT_).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\exp_1500.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25
set RWKV_EMPTY_CACHE_EVERY=0

echo ===== EXP 1500 (1 epoch WS on 1000-2499) START %DATE% %TIME% ===== > "%LOG%"
echo === WS 1 epoch %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/train_1500_ws.toml >> "%LOG%" 2>&1
del /Q result\RWKV-t1500.jsonl result\RWKV-P-t1500.jsonl 2>nul
echo === WRITE EVAL TOML (find latest ckpt) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/exp_1500 t1500ws scratchpad/eval_1500.toml RWKV-t1500 RWKV-P-t1500 >> "%LOG%" 2>&1
echo === EVAL 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/eval_1500.toml >> "%LOG%" 2>&1
echo === SCORE (vs champion 0.314807/0.280200) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-t1500 RWKV-P-t1500 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
