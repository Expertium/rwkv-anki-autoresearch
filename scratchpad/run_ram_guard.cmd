@echo off
REM Keep free RAM above 14 GB for the duration of long unattended training.
REM WHY: 2026-07-31 01:32 the tuner's two fetch workers had grown to 24.75 + 24.05 GB of
REM LMDB mmap pages, leaving 0.7 GB free -- i.e. INSIDE the 56-63 GB band that preceded all
REM three unexplained black-screen hangs, and deeper into it than any of them. A single
REM EmptyWorkingSet pass reclaimed 46.6 GB (1.0 -> 47.6 GB free) with no effect on the run.
REM Measured rate at MAX=65536 with NUM_FETCH_PROCESSES=2: ~38 GB/h, so from ~50 GB free the
REM box re-enters the band in ~1.2 h and EVERY WS phase (~2.5 h) would enter it unguarded.
REM Non-destructive: the pages are CLEAN and file-backed, so Windows drops them and re-faults
REM on demand. Costs a brief slowdown; prevents a hang that would cost a 4+ h trial.
REM
REM ⚠ TWO BUGS FIXED 2026-08-02, both found the hard way:
REM  1. It USED TO EXIT after -Minutes 2880 (48 h) and die SILENTLY while training continued.
REM     The HP tuner outlived it. Now wrapped in an infinite restart loop, so the guard's
REM     lifetime is bounded by the machine, not by a constant.
REM  2. The log was opened with ">" which TRUNCATES -- restarting the guard destroyed the
REM     record of every trim it had done. Now ">>" (append), with a start banner.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
echo ===== RAM GUARD START %DATE% %TIME% ===== >> scratchpad\ram_guard.log
:loop
powershell -NoProfile -ExecutionPolicy Bypass -File scratchpad\ram_guard.ps1 -FloorGB 14 -IntervalSec 60 -Minutes 1440 >> scratchpad\ram_guard.log 2>&1
echo [guard] inner run ended (exit %ERRORLEVEL%) %DATE% %TIME% - restarting >> scratchpad\ram_guard.log
timeout /t 5 /nobreak > nul
goto loop
