@echo off
REM H=2/K=16 (2x-smaller-state) experiment on the 1500-user champion recipe: WS (1 epoch, 1000-2499) +
REM 0.27-epoch decay -> eval 101-200 -> score vs the 1500-data champion (0.309706/0.276357). Same recipe,
REM ONLY the arch differs (RWKV_N_HEADS=2 RWKV_HEAD_DIM=16 -> d=32, K=16, per-card state 1088->576).
REM Detached. Monitor scratchpad/exp_h2k16.log (DONE_EXIT_). The K=16 CUDA kernel is parity-verified.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\exp_h2k16.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16

echo ===== H2K16 START %DATE% %TIME% ===== > "%LOG%"
echo === WS 1 epoch (1000-2499) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/train_h2k16_ws.toml >> "%LOG%" 2>&1
echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/exp_h2k16 h2k16ws h2k16d scratchpad/train_h2k16_decay.toml train_db_sc8k_1500 1000 2499 0.27 >> "%LOG%" 2>&1
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/train_h2k16_decay.toml >> "%LOG%" 2>&1
del /Q result\RWKV-h2k16.jsonl result\RWKV-P-h2k16.jsonl 2>nul
echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/exp_h2k16 h2k16d scratchpad/eval_h2k16.toml RWKV-h2k16 RWKV-P-h2k16 >> "%LOG%" 2>&1
echo === EVAL 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/eval_h2k16.toml >> "%LOG%" 2>&1
echo === SCORE (vs 1500 champion 0.309706/0.276357) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-h2k16 RWKV-P-h2k16 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
