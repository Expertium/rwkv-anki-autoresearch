@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner\hp_weight_decay_0p1.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_WEIGHT_DECAY=0.1
set RWKV_CLIP=0.5
echo ===== TRIAL hp_weight_decay_0p1 (param=weight_decay=0.1) START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner/hp_weight_decay_0p1_ws.toml >> "%LOG%" 2>&1
del /Q result\RWKV-hp_weight_decay_0p1.jsonl result\RWKV-P-hp_weight_decay_0p1.jsonl 2>nul
echo ===== EVAL hp_weight_decay_0p1 %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/tuner/hp_weight_decay_0p1_eval.toml >> "%LOG%" 2>&1
echo ===== RECORD hp_weight_decay_0p1 %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe optimization/hp_tuner.py record hp_weight_decay_0p1 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
