@echo off
REM Wait for the learnable-codebook eval to finish, then run the imm-independence check
REM (dump our champion over the teacher's batch stream, then compare offline).
REM
REM ⚠ cblearn.log is APPENDED across runs, so a terminal token from the aborted first attempt sits
REM in the file. Waiting on it directly fires instantly (that trap cost a false "run finished" on
REM 2026-08-13). This waits on the RESULT ARTIFACT instead -- the merged VAL-half jsonl, which
REM exists only when this run's eval actually completed.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\wait_immcorr.log
set DONEFILE=C:\Users\Andrew\rwkv-anki-autoresearch\result\RWKV-P-qtaxd_cblearn.jsonl
echo waiting for the cblearn VAL-half eval %DATE% %TIME% > "%LOG%"

:wait
if exist "%DONEFILE%" goto ready
ping -n 61 127.0.0.1 >nul
goto wait

:ready
ping -n 61 127.0.0.1 >nul
echo cblearn eval done, starting the immcorr dump %DATE% %TIME% >> "%LOG%"
call scratchpad\qat_tax\run_immcorr_dump.cmd
echo dump returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
.venv\Scripts\python.exe -u scratchpad/qat_tax/imm_independence.py >> "%LOG%" 2>&1
echo analysis returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo IMMCORRCHAIN_EXIT_0 %DATE% %TIME% >> "%LOG%"
