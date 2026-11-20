@echo off
REM TA3 4-epoch chain, RESTART after the per-group A/Q split fix (2026-11-16).
REM Controller: scratchpad/ta3/ta3_train.py
REM   chain log -> scratchpad/ta3/chain_20261116.log
REM   run dir   -> scratchpad/ta3_fullrun_20261116
REM The first attempt (scratchpad/ta3_fullrun_20261115, chain.log) aborted at epoch-1 step 4930 and is
REM preserved, not resumed. This restart begins epoch 1 from the QAT46 step-248 pair.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u scratchpad\ta3\ta3_train.py --run-dir scratchpad\ta3_fullrun_20261116 --epochs 4 --retries 2 > scratchpad\ta3\chain_20261116.log 2>&1
set RC=%ERRORLEVEL%
(echo TA3_CHAIN_EXIT_%RC%)>> scratchpad\ta3\chain_20261116.log
exit /b %RC%
