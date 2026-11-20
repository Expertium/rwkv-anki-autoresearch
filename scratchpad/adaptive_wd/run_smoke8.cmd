@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
.venv\Scripts\python.exe -u scratchpad\adaptive_wd\adaptive_train.py --smoke --smoke-from-scratch --run-dir scratchpad\adaptive_wd\smoke8 --max-attempts 30 > scratchpad\adaptive_wd\smoke8\smoke.log 2>&1
set RC=%ERRORLEVEL%
(echo SMOKE_EXIT_%RC%)>> scratchpad\adaptive_wd\smoke8\smoke.log
exit /b %RC%
