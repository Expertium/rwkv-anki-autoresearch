@echo off
REM Full read-only scan of the training LMDB label layouts to prove the fixed TA3 split never
REM leaves a phase empty (2026-11-16). CPU only, no GPU.
REM   log -> scratchpad/ta3/layout_scan.log
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u scratchpad\ta3\scan_batch_layouts.py --workers 4 --users 1-5000 > scratchpad\ta3\layout_scan.log 2>&1
echo SCAN_EXIT_%ERRORLEVEL% %DATE% %TIME% >> scratchpad\ta3\layout_scan.log
