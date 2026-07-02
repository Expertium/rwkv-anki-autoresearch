@echo off
REM Batch-size/throughput sweep (Andrew): find the MAX_TRAIN_GLOBAL_LEN that maxes GPU training
REM throughput without OOMing 12 GB. Detached via detach.ps1 (survives Esc). Monitor scratchpad/batch_sweep.log.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\batch_sweep.log
echo ===== BATCH SWEEP START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad\batch_sweep.py >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
