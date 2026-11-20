@echo off
REM TA3 4-epoch chain launcher (2026-11-15). Controller: scratchpad/ta3/ta3_train.py
REM   chain log -> scratchpad/ta3/chain.log
REM   run dir   -> scratchpad/ta3_fullrun_20261115
REM Start the run ONLY after the corrected 0.04-epoch smoke in
REM scratchpad/ta3_smoke_fixed_20261115/ has finished clean.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u scratchpad\ta3\ta3_train.py --run-dir scratchpad\ta3_fullrun_20261115 --epochs 4 > scratchpad\ta3\chain.log 2>&1
set RC=%ERRORLEVEL%
(echo TA3_CHAIN_EXIT_%RC%)>> scratchpad\ta3\chain.log
exit /b %RC%
