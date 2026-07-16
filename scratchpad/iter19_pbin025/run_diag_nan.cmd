@echo off
REM NaN layer diagnosis for iter19 user 8902 (Andrew 2026-07-16). fp32, NO_JIT (hooks).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter19_pbin025\diag_nan.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_NO_JIT=1
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_ZERO_FEATURES=22
echo ===== DIAG_NAN START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad\iter19_pbin025\diag_nan_layer.py >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
