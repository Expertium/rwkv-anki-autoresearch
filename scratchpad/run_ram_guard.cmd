@echo off
REM Keep free RAM above 14 GB for the duration of the HP tuner (~40 h remaining).
REM WHY: 2026-07-31 02:26 the tuner's two fetch workers had grown to 24.75 + 24.05 GB of
REM LMDB mmap pages, leaving 0.7 GB free -- i.e. INSIDE the 56-63 GB band that preceded all
REM three unexplained black-screen hangs, and deeper into it than any of them. A single
REM EmptyWorkingSet pass reclaimed 46.6 GB (1.0 -> 47.6 GB free) with no effect on the run.
REM NUM_FETCH_PROCESSES=2 halved the climb rate but did not remove it.
REM Non-destructive: the pages are CLEAN and file-backed, so Windows drops them and re-faults
REM on demand. Costs a brief slowdown; prevents a hang that would cost a 4 h trial.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
powershell -NoProfile -ExecutionPolicy Bypass -File scratchpad\ram_guard.ps1 -FloorGB 14 -IntervalSec 60 -Minutes 2880 > scratchpad\ram_guard.log 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> scratchpad\ram_guard.log
