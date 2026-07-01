@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner\hp_clip_1.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=1
echo ===== TRIAL hp_clip_1 (param=clip=1) START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner/hp_clip_1_ws.toml >> "%LOG%" 2>&1
del /Q result\RWKV-hp_clip_1.jsonl result\RWKV-P-hp_clip_1.jsonl 2>nul
echo ===== EVAL hp_clip_1 %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/tuner/hp_clip_1_eval.toml >> "%LOG%" 2>&1
echo ===== RECORD hp_clip_1 %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe optimization/hp_tuner.py record hp_clip_1 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
