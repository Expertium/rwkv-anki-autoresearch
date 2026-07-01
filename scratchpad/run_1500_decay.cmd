@echo off
REM Decay phase for the 1500-user experiment: ~0.27-epoch cosine decay (SAME WS:decay ratio as the
REM champion's 15:4 -> ~900 steps, a comparable annealing tail) from the 1500 WS-final, eval 101-200,
REM score vs champion (0.314807/0.280200). Run AFTER run_train_1500.cmd finishes. Detached. Monitor
REM scratchpad/exp_1500_decay.log (DONE_EXIT_).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\exp_1500_decay.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25
set RWKV_EMPTY_CACHE_EVERY=0

echo ===== EXP 1500 DECAY START %DATE% %TIME% ===== > "%LOG%"
echo === SETUP (find WS-final, rename optim, write decay toml) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/exp_1500 t1500ws t1500d scratchpad/train_1500_decay.toml train_db_sc8k_1500 1000 2499 0.27 >> "%LOG%" 2>&1
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/train_1500_decay.toml >> "%LOG%" 2>&1
del /Q result\RWKV-t1500d.jsonl result\RWKV-P-t1500d.jsonl 2>nul
echo === WRITE EVAL TOML (find latest decay ckpt) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/exp_1500 t1500d scratchpad/eval_1500d.toml RWKV-t1500d RWKV-P-t1500d >> "%LOG%" 2>&1
echo === EVAL 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/eval_1500d.toml >> "%LOG%" 2>&1
echo === SCORE (vs champion 0.314807/0.280200) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-t1500d RWKV-P-t1500d >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
