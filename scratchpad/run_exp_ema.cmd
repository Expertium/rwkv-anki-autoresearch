@echo off
REM EMA experiment: WS-15 with EMA weight-averaging (decay 0.999), eval the AVERAGED weights (no decay phase)
REM vs the champion WS-15+decay (0.314807/0.280200). Tests whether averaging rivals/replaces the decay phase
REM (lit 2026-06-30). Detached. Monitor scratchpad/exp_ema.log (DONE_EXIT_). EARLY CHECK: emaws_ema_800.pth
REM should appear ~22 min in; if missing, the EMA save path is broken.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\exp_ema.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25
set RWKV_EMA_DECAY=0.999

echo ===== EXP ema (WS-15 + EMA0.999, no decay) START %DATE% %TIME% ===== > "%LOG%"
echo === WS-15 + EMA %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/ema_ws.toml >> "%LOG%" 2>&1
del /Q result\RWKV-ema.jsonl result\RWKV-P-ema.jsonl 2>nul
echo === EVAL (averaged weights) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/ema_eval.toml >> "%LOG%" 2>&1
echo === SCORE (vs champion 0.314807/0.280200) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-ema RWKV-P-ema >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
