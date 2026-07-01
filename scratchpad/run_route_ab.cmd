@echo off
REM Route-R follow-ups (a) accuracy confirm/sweep + (b) higher-MAX speed. Builds the 4096-chunk db,
REM then runs the 4-run orchestrator (sc8k 2nd seed, sc4k, sc8k@MAX132000, sc8k@MAX200000) + evals on
REM 101-200 + summary. Detached (Esc/teardown-proof); monitor scratchpad/route_ab.log for DONE_EXIT.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\route_ab.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1

echo ================ ROUTE A/B START %DATE% %TIME% ================ > "%LOG%"
echo === build sc4k db (4096-chunk) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_config_sc4k.toml >> "%LOG%" 2>&1
echo === orchestrator (4 runs + evals) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/route_ab.py >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
