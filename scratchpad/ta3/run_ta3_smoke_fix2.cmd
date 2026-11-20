@echo off
REM TA3 0.04-epoch smoke #2, after the per-group A/Q split fix (2026-11-16).
REM   launcher -> scratchpad/ta3/ta3_smoke_fix2_launcher.log
REM   trainer  -> scratchpad/ta3_smoke_fix2_20261116/train.log
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u scratchpad\ta3\mk_ta3_smoke_fix2.py > scratchpad\ta3\ta3_smoke_fix2_launcher.log 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> scratchpad\ta3\ta3_smoke_fix2_launcher.log
