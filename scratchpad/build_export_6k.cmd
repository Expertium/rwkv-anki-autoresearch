@echo off
REM Chained: (1) build label_filter_db for held-out users 6000-6999 (find_equalize, APPENDS, resumable,
REM IDLE priority), then (2) export those 1000 feature traces to the sibling quant folder. Detached
REM (survives Esc/teardown). Monitor scratchpad/build_export_6k.log (BUILD_DONE_, then DONE_EXIT_).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\build_export_6k.log
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch;C:\Users\Andrew
set OMP_NUM_THREADS=8
echo ===== BUILD label_filter_db 6000-6999 %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -m rwkv.find_equalize_test_reviews --config rwkv\find_equalize_6k_config.toml >> "%LOG%" 2>&1
echo ===== BUILD_DONE_%ERRORLEVEL% %DATE% %TIME% ===== >> "%LOG%"
echo ===== EXPORT 6000-6999 to sibling reference_big/ %DATE% %TIME% ===== >> "%LOG%"
set RWKV_TRACE_OUT=C:\Users\Andrew\rwkv-state-quant\reference_big
.venv\Scripts\python.exe scratchpad\export_mp.py --procs 8 --range 6000 7000 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
