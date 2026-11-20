@echo off
REM Lean single-process scan of the training LMDB label layouts (2026-11-16). Read-only, CPU only.
REM A 4-worker pool hung on 2026-11-16 (~20 min, workers busy, no results) -- see the module docstring.
REM   log -> scratchpad/ta3/layout_scan_lean.log
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u scratchpad\ta3\scan_batch_layouts.py --users 1-5000 > scratchpad\ta3\layout_scan_lean.log 2>&1
echo SCAN_EXIT_%ERRORLEVEL% %DATE% %TIME% >> scratchpad\ta3\layout_scan_lean.log
