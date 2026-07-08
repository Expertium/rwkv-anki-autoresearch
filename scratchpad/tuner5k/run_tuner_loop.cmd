@echo off
REM hp_tuner_5k self-driving loop (coordinate descent, resumable via journal replay).
REM Each trial .cmd it spawns is self-contained + self-recording. Launch DETACHED.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u optimization\hp_tuner_5k.py loop >> scratchpad\tuner5k\loop.log 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> scratchpad\tuner5k\loop.log
