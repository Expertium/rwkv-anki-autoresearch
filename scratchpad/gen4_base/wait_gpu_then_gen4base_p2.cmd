@echo off
REM =========================================================================================
REM Park gen4base PHASE 2 behind Andrew's srs-benchmark GRU pretrain (2026-09-02 21:35).
REM The decay deadlocked in WDDM paging beside it: 8.7 GB (decay alone) + the GRU job pinned VRAM
REM at 11.7 of 12.28 GB, 100 pct util, ZERO steps for 16 min (steps 485 at 21:18, nothing after).
REM Rule: no co-tenant GPU work during a gate-critical run, and his job is his -- so this waits
REM until no reptile_trainer_gru process exists, then calls run_gen4base_p2.cmd unchanged.
REM Andrew can override by stopping the GRU job; this fires within 2 minutes of that.
REM The probe counts PYTHON processes only: the probe's own powershell line and any bash shell that
REM typed the pattern also match it, and without the Name filter the count could never reach 0
REM (caught by executing the probe before arming: it returned 6, then 5).
REM No angle brackets or arrows in REM lines.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\wait_gpu.log
set NGRU=1

echo ===== gpu-free waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitgpu
for /f %%R in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'reptile_trainer_gru' -and $_.Name -eq 'python.exe' } | Measure-Object).Count"') do set NGRU=%%R
if not "%NGRU%"=="0" (
  ping -n 121 127.0.0.1 >nul
  goto waitgpu
)
echo GPU free of the GRU pretrain %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\run_gen4base_p2.cmd
echo run_gen4base_p2 returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
