@echo off
REM Daily read-only commit check for JSchoreels/anki-rwkv.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u scratchpad\anki_rwkv_peer_watch.py >> scratchpad\anki_rwkv_peer_watch.out 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> scratchpad\anki_rwkv_peer_watch.out
