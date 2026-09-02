@echo off
REM =========================================================================================
REM Start the GENERATION 5 rebuild (real-time cycles) once gen4base is into its EVAL phase.
REM
REM WHY DECAY_OK AND NOT THE TERMINAL MARKER. The realcyc run needs the GPU after gen4base and
REM after the queued feature ablation (~1.5 h), so gen 5 must be READY by then -- about 3.5 h of
REM CPU. gen4base's eval is GPU-bound with two light fetch workers, so a CPU rebuild beside it
REM costs little; beside its WS or decay it would slow a dispatch-bound trainer. DECAY_OK is
REM the earliest moment that is both safe and early enough.
REM
REM A terminal marker is ALSO accepted, so a gen4base that dies before decay does not strand
REM gen 5 forever; and the RAM check stays, because a marker alone is satisfied by a dead-and-
REM relaunched gen4base (the lesson from wait_then_rebuild4).
REM
REM Anchored findstr -- the unanchored form matches this file's own prose and fires instantly.
REM No angle brackets or arrows in REM lines.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\wait_rebuild5.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\gen4base.log
set MINFREEMB=25000

echo ===== gen5 waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"gen4base DECAY_OK" "%PREVLOG%" >nul 2>&1
if not errorlevel 1 goto ready
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if not errorlevel 1 goto ready
ping -n 121 127.0.0.1 >nul
goto waitprev

:ready
echo gen4base reached decay-done or terminal %DATE% %TIME% >> "%LOG%"

:waitram
for /f %%R in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"') do set FREEMB=%%R
if %FREEMB% LSS %MINFREEMB% (
  echo waiting for RAM: %FREEMB% MB free, need %MINFREEMB% MB  %DATE% %TIME% >> "%LOG%"
  ping -n 121 127.0.0.1 >nul
  goto waitram
)
echo RAM OK: %FREEMB% MB free %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\run_rebuild5.cmd
echo rebuild5 returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
