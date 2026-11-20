@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
.venv\Scripts\python.exe -u scratchpad\adaptive_wd\adaptive_train.py --smoke --smoke-from-scratch --run-dir scratchpad\adaptive_wd\smoke2 --max-attempts 30 > scratchpad\adaptive_wd\smoke2\smoke.log 2>&1
set RC=%ERRORLEVEL%
(echo SMOKE_EXIT_%RC%)>> scratchpad\adaptive_wd\smoke2\smoke.log
exit /b %RC%
