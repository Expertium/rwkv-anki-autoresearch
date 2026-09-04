@echo off
REM Detached wrapper for the SAM CPU validation (three arms, ~20-30 min). Log: validate_cpu.log
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sam\validate_cpu.log
echo ===== SAM CPU VALIDATE START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad/sam/validate_cpu.py >> "%LOG%" 2>&1
echo validate rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
