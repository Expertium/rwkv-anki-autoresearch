@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner5k\hp5k_baseline.log
set OMP_NUM_THREADS=5
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
echo ===== TRIAL hp5k_baseline (param=baseline=baseline) cfg={"peak_lr": 0.001, "warmup_steps": 200, "weight_decay": 0.01, "clip": 0.25} START %DATE% %TIME% ===== > "%LOG%"
echo === WS 2 epochs (1000-2499) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/hp5k_baseline/hp5k_baseline_ws.toml >> "%LOG%" 2>&1
echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/tuner5k/hp5k_baseline hp5k_baselinews hp5k_baselined scratchpad/tuner5k/hp5k_baseline/hp5k_baseline_decay.toml train_db_sc8k_1500 1000 2499 0.5 0.001 >> "%LOG%" 2>&1
echo === DECAY 0.5 epoch %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/hp5k_baseline/hp5k_baseline_decay.toml >> "%LOG%" 2>&1
del /Q result\RWKV-hp5k_baseline.jsonl result\RWKV-P-hp5k_baseline.jsonl 2>nul
echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/tuner5k/hp5k_baseline hp5k_baselined scratchpad/tuner5k/hp5k_baseline/hp5k_baseline_eval.toml RWKV-hp5k_baseline RWKV-P-hp5k_baseline >> "%LOG%" 2>&1
echo === EVAL 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/tuner5k/hp5k_baseline/hp5k_baseline_eval.toml >> "%LOG%" 2>&1
echo === RECORD hp5k_baseline %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/hp_tuner_5k.py record hp5k_baseline >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
