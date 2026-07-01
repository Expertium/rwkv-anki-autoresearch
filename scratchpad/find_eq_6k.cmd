@echo off
REM Build label_filter_db entries for the 500+500 held-out quant users (6000-6999). Appends (skips
REM existing). Detached via detach.ps1 (survives Esc/teardown). Monitor scratchpad/find_eq_6k.log
REM (DONE_EXIT_). Low-priority workers (won't fight the FSRS benchmark). map_size 2GB (export-compatible).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\find_eq_6k.log
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch;C:\Users\Andrew
set OMP_NUM_THREADS=8
set PYTHONUNBUFFERED=1
echo ===== FIND_EQ 6000-6999 START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.find_equalize_test_reviews --config rwkv/find_equalize_6k_config.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
