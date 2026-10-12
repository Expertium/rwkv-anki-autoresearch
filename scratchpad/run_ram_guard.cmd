@echo off
REM RAM guard (Andrew 2026-09-15: keep at least 10 GB of RAM free). Caps every rwkv fetch worker's working set
REM at 6-16 GB (dynamic: grows while RAM is spare, shrinks under the 10 GB floor) so the LMDB file pages return to the cache. Detach via scratchpad/detach.ps1; log = ram_guard.log.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
set RWKV_RAM_GUARD_CAP_GB=6
set RWKV_RAM_GUARD_FLOOR_GB=10
.venv\Scripts\python.exe -u scratchpad/ram_guard.py >> scratchpad\ram_guard.out 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> scratchpad\ram_guard.out
