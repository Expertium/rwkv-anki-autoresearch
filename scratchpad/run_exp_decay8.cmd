@echo off
REM EXP decay-length: WS-15 + 8-epoch decay (champion uses 4). Cheap (~15 min, no WS retrain). Gated vs the
REM 4-epoch decay champion (0.314807/0.280200). Detached. Monitor scratchpad/exp_decay8.log (DONE_EXIT_).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\exp_decay8.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25

echo ===== EXP decay8 (WS-15 + 8ep decay) START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/exp_decay8.toml >> "%LOG%" 2>&1
del /Q result\RWKV-decay8.jsonl result\RWKV-P-decay8.jsonl 2>nul
echo === EVAL %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/exp_decay8_eval.toml >> "%LOG%" 2>&1
echo === SCORE (vs 4ep-decay champion 0.314807/0.280200) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-decay8 RWKV-P-decay8 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
