@echo off
REM Re-run the imm-independence check after the norm probe finishes. The first attempt died on a
REM toml the generator had corrupted (a Windows path in a heredoc header wrote literal CR/TAB);
REM the toml is fixed and the dump runner now gates on its exit code.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set NLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\wait_normprobe.log
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\wait_immcorr2.log
echo waiting for the norm probe %DATE% %TIME% > "%LOG%"

:wait
findstr /B /C:"NORMCHAIN_EXIT_" "%NLOG%" >nul
if %ERRORLEVEL%==0 goto ready
ping -n 61 127.0.0.1 >nul
goto wait

:ready
ping -n 31 127.0.0.1 >nul
echo norm probe done, re-running the immcorr dump %DATE% %TIME% >> "%LOG%"
call scratchpad\qat_tax\run_immcorr_dump.cmd
if not %ERRORLEVEL%==0 (
  echo DUMP FAILED -- skipping analysis %DATE% %TIME% >> "%LOG%"
  echo IMMCORR2_EXIT_1 >> "%LOG%"
  exit /b 1
)
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
.venv\Scripts\python.exe -u scratchpad/qat_tax/imm_independence.py >> "%LOG%" 2>&1
echo analysis returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo IMMCORR2_EXIT_0 %DATE% %TIME% >> "%LOG%"
