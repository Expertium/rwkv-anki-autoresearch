@echo off
REM Full-dataset REVIEWED-entity census (all 9,934 users). CPU + disk only.
REM 6 workers, not more: the QAT-tax chain holds the GPU and its eval fetch workers need CPU.
REM The dataset is on C: while the eval's LMDB is on F:, so disk contention is limited.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\reviewed_counts.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1

echo ===== REVIEWED COUNTS START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad/reviewed_entity_counts.py scratchpad/reviewed_entity_counts_10k.csv --workers 6 >> "%LOG%" 2>&1
echo CENSUS_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"
echo CENSUSDONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
