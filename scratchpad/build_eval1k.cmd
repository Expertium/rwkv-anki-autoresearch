@echo off
REM Groundwork for step 4 (Andrew 2026-06-29): build preprocessed data for users 1001-2000 (the
REM step-4 TEST set). PHASE1 = find_equalize (label_filter_db, CPU). PHASE2 = data_processing
REM (test_db features, GPU). Both APPEND (skip already-done users incl the 1001-1003 smoke).
REM Detached (WMI/WmiPrvSE) so it survives Esc/teardown. Monitor scratchpad/build_eval1k.log.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\build_eval1k.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
echo === PHASE1 find_equalize 1001-2000 START %DATE% %TIME% === > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.find_equalize_test_reviews --config rwkv/find_equalize_eval1k_config.toml >> "%LOG%" 2>&1
echo === PHASE1 DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
echo === PHASE2 data_processing test 1001-2000 START %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_config_eval1k.toml >> "%LOG%" 2>&1
echo === PHASE2 DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
