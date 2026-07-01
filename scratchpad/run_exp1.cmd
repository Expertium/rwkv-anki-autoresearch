@echo off
REM EXP1 (task-5 arch experiment): num_curves/num_points 64->128 (restore SRS-head resolution the champion
REM gave up for params; now affordable under the 225k cap, 209,312 params, ZERO state cost). Full tuned recipe:
REM WS-15 + 4-epoch decay (peak_lr 1e-3 / warmup 200 / clip 0.25), eval 101-200, gated vs the decay champion.
REM Detached (Esc/teardown-proof). Monitor scratchpad/exp1.log (DONE_EXIT_).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\exp1.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25
set RWKV_NUM_CURVES=128
set RWKV_NUM_POINTS=128

echo ===== EXP1 (curves/points 128) START %DATE% %TIME% ===== > "%LOG%"
echo === WS-15 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/exp1_ws.toml >> "%LOG%" 2>&1
copy /Y scratchpad\exp1\exp1ws_optim_2400.pth scratchpad\exp1\exp1ws_2400_optim.pth >> "%LOG%" 2>&1
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/exp1_decay.toml >> "%LOG%" 2>&1
del /Q result\RWKV-exp1.jsonl result\RWKV-P-exp1.jsonl 2>nul
echo === EVAL %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/exp1_eval.toml >> "%LOG%" 2>&1
echo === SCORE (vs decay champion 0.314807/0.280200) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-exp1 RWKV-P-exp1 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
