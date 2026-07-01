@echo off
REM WSD decay phase on the tuned WS-15 champion: copy the WS-15 optimizer into the loader's expected
REM {NAME}_optim.pth form, run a 4-epoch cosine decay (LR 1e-3 -> 0), eval on 101-200, score vs the
REM WS-15 champion + d128 baseline. Detached (Esc/teardown-proof); monitor scratchpad/tuner/decay15.log.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner\decay15.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_CLIP=0.25

echo ===== DECAY15 START %DATE% %TIME% ===== > "%LOG%"
copy /Y scratchpad\tuner\hp_epochs_15\hp_epochs_15_optim_2400.pth scratchpad\tuner\hp_epochs_15\hp_epochs_15_2400_optim.pth >> "%LOG%" 2>&1
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner_decay15.toml >> "%LOG%" 2>&1
del /Q result\RWKV-decay15.jsonl result\RWKV-P-decay15.jsonl 2>nul
echo ===== EVAL decay15 %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/tuner_decay15_eval.toml >> "%LOG%" 2>&1
echo ===== SCORE decay15 %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe scratchpad/score_jsonl.py RWKV-decay15 RWKV-P-decay15 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
