@echo off
REM Re-baseline the champion at FULL coverage (66000): train the WS config TWICE from scratch (run1,
REM run2) -> gives BOTH the run-to-run variance (run1 vs run2) AND a fair full-coverage champion (run1).
REM Detached (survives Esc/teardown). Evals are done separately (interactively) to avoid the GPU-
REM contention eval-write failure seen before. RUN ONLY AFTER the data build (build_eval1k) finishes.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\rebaseline.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
echo === RUN1 WS 66000 (from scratch, full coverage) START %DATE% %TIME% === > "%LOG%"
if exist scratchpad\rebase_run rmdir /S /Q scratchpad\rebase_run
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/rebase_66k_ws.toml >> "%LOG%" 2>&1
echo === RUN1 train exit %ERRORLEVEL% %TIME% === >> "%LOG%"
if exist scratchpad\rebase_run1 rmdir /S /Q scratchpad\rebase_run1
move /Y scratchpad\rebase_run scratchpad\rebase_run1 >> "%LOG%" 2>&1
echo === RUN2 WS 66000 START %TIME% === >> "%LOG%"
if exist scratchpad\rebase_run rmdir /S /Q scratchpad\rebase_run
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/rebase_66k_ws.toml >> "%LOG%" 2>&1
echo === RUN2 train exit %ERRORLEVEL% %TIME% === >> "%LOG%"
if exist scratchpad\rebase_run2 rmdir /S /Q scratchpad\rebase_run2
move /Y scratchpad\rebase_run scratchpad\rebase_run2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
