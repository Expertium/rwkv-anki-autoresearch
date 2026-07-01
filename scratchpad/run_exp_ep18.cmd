@echo off
REM EPOCHS experiment: WS-18 + 4-epoch decay (vs champion WS-15+decay). Tests whether more than 15 WS epochs
REM still helps (epochs was the 2nd-biggest tuner lever, not saturated at 15). Gated vs the champion
REM (0.314807/0.280200). Detached. Monitor scratchpad/exp_ep18.log (DONE_EXIT_).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\exp_ep18.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25

echo ===== EXP ep18 (WS-18 + decay) START %DATE% %TIME% ===== > "%LOG%"
echo === WS-18 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/ep18_ws.toml >> "%LOG%" 2>&1
copy /Y scratchpad\exp_ep18\ep18ws_optim_2880.pth scratchpad\exp_ep18\ep18ws_2880_optim.pth >> "%LOG%" 2>&1
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/ep18_decay.toml >> "%LOG%" 2>&1
del /Q result\RWKV-ep18.jsonl result\RWKV-P-ep18.jsonl 2>nul
echo === EVAL %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/ep18_eval.toml >> "%LOG%" 2>&1
echo === SCORE (vs champion 0.314807/0.280200) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-ep18 RWKV-P-ep18 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
