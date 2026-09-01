@echo off
REM =========================================================================================
REM Start the GENERATION 4 rebuild once featB is out of the way AND the RAM is actually there.
REM
REM WHY IT WAITS AT ALL. The rebuild is CPU-only, so the obvious move is to run it beside the
REM training the way generation 2 ran beside featA. That precedent does not transfer: featA's
REM fetch workers were about 9.7 GB, featB's measured 18.3 GB each on 2026-09-01, and free RAM
REM was 10.9 GB of 63.9. Gen 3's own config header records a rebuild exhausting 64 GB and dying
REM beside a training run. Starting early buys nothing -- nothing consumes gen 4 until featB
REM reports -- and risks a 10 h run. So it waits.
REM
REM TWO CONDITIONS, NOT ONE. A terminal marker alone is not enough: if featB dies and gets
REM relaunched, the marker is present while a fresh run is starting. The RAM check is what makes
REM this safe against that, and it is the same condition that motivated the wait.
REM
REM It does NOT require featB to have SUCCEEDED. Andrew wants the id fixes regardless of how the
REM features arm turns out, so the rebuild is not conditional on the experiment's verdict.
REM
REM Anchored findstr -- the unanchored form matches this file's own prose and fires instantly.
REM No angle brackets or arrows in REM lines.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\wait_rebuild4.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featB\featB.log
set MINFREEMB=25000

echo ===== gen4 waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo featB reported %DATE% %TIME% >> "%LOG%"

:waitram
for /f %%R in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"') do set FREEMB=%%R
if %FREEMB% LSS %MINFREEMB% (
  echo waiting for RAM: %FREEMB% MB free, need %MINFREEMB% MB  %DATE% %TIME% >> "%LOG%"
  ping -n 121 127.0.0.1 >nul
  goto waitram
)
echo RAM OK: %FREEMB% MB free %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\run_rebuild4.cmd
echo rebuild4 returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
