@echo off
REM TA3 corrected 0.04-epoch smoke launcher (2026-11-15). Builds the smoke config from the REAL
REM start checkpoint (QAT46 step-248) and launches the trainer with RWKV_TA3=1. Logs:
REM   launcher -> scratchpad/adaptive_wd/ta3_smoke_fixed_launcher.log
REM   trainer  -> scratchpad/ta3_smoke_fixed_20261115/train.log
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u scratchpad\adaptive_wd\mk_ta3_smoke_fixed.py > scratchpad\adaptive_wd\ta3_smoke_fixed_launcher.log 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> scratchpad\adaptive_wd\ta3_smoke_fixed_launcher.log
