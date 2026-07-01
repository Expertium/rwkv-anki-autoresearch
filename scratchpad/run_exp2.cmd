@echo off
REM EXP2 (task-5 arch experiment): channel_mixer_factor 1.0->1.5 (per-block FFN capacity; the d=128 model
REM used 1.5-2.0; our d=32 is capacity-starved). 207,136 params (<=225k), ZERO state cost. Full tuned recipe:
REM WS-15 + 4-epoch decay, eval 101-200, gated vs the decay champion. Detached. Monitor scratchpad/exp2.log.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\exp2.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25
set RWKV_CHANNEL_MIXER_FACTOR=1.5

echo ===== EXP2 (channel_mixer 1.5) START %DATE% %TIME% ===== > "%LOG%"
echo === WS-15 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/exp2_ws.toml >> "%LOG%" 2>&1
copy /Y scratchpad\exp2\exp2ws_optim_2400.pth scratchpad\exp2\exp2ws_2400_optim.pth >> "%LOG%" 2>&1
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/exp2_decay.toml >> "%LOG%" 2>&1
del /Q result\RWKV-exp2.jsonl result\RWKV-P-exp2.jsonl 2>nul
echo === EVAL %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/exp2_eval.toml >> "%LOG%" 2>&1
echo === SCORE (vs decay champion 0.314807/0.280200) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-exp2 RWKV-P-exp2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
