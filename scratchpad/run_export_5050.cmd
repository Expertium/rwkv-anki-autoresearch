@echo off
REM Export RWKV per-review trace inputs for users 101-200 (the 50+50 dev/val split for the outsourced
REM state-quant loop) STRAIGHT INTO the sibling folder's reference/ (skips the 17 already there).
REM CPU-only feature extraction, resumable (skips existing trace_user_*.safetensors). Detached. Monitor
REM scratchpad/export_5050.log (DONE_EXIT_). Reads ../anki-revlogs-10k + label_filter_db (concurrent-safe).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\export_5050.log
set OMP_NUM_THREADS=3
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_TRACE_OUT=C:\Users\Andrew\rwkv-state-quant\reference
echo ===== EXPORT 5050 (traces 101-200 -> sibling reference) START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad/export_features_fast.py --range 101 201 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
