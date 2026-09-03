@echo off
REM gen4base EVAL RESUME, attempt 3 (2026-09-03 08:15). User 6701's million-row eval chunks need ~42 GB of
REM GPU-addressable memory; under WDDM the excess is SYSTEM RAM, and both failures (04:50, 08:06) had the
REM gen-5 rebuild eating it (58 of 64 GB used at 08:06). Gate: rebuild5 SUCCESS plus 45 GB free RAM.
REM The runner appends to gen4base_evalresume.log; success there is the line gen4base EVAL_OK.
REM No percent-tilde in REM lines.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\wait_evalresume2.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\rebuild5.log
set MINFREEMB=45000
set FREEMB=0

echo ===== evalresume2 waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitall
findstr /B /C:"DONE_EXIT_0" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitall
)
for /f %%R in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"') do set FREEMB=%%R
if %FREEMB% LSS %MINFREEMB% (
  echo waiting for RAM: %FREEMB% MB free, need %MINFREEMB% MB %DATE% %TIME% >> "%LOG%"
  ping -n 121 127.0.0.1 >nul
  goto waitall
)
echo RAM OK: %FREEMB% MB free %DATE% %TIME% >> "%LOG%"
echo all gates open %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\run_gen4base_evalresume.cmd
echo run_gen4base_evalresume returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
