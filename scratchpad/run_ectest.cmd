@echo off
REM empty_cache A/B: run the SAME short WS twice -- RWKV_EMPTY_CACHE_EVERY=1 (baseline, byte-identical
REM to old behavior) vs =0 (off) -- compare steps/s + train_elapsed_min, watch for OOM. Detached.
REM Monitor scratchpad/ectest.log (DONE_EXIT_). RUN ONLY AFTER build_1500 finishes (else CPU contention).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ectest.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234

echo ===== EC TEST START %DATE% %TIME% ===== > "%LOG%"
echo === RUN A: RWKV_EMPTY_CACHE_EVERY=1 (baseline) %TIME% === >> "%LOG%"
set RWKV_EMPTY_CACHE_EVERY=1
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/ec_test_ws.toml >> "%LOG%" 2>&1
echo === RUN B: RWKV_EMPTY_CACHE_EVERY=0 (off) %TIME% === >> "%LOG%"
set RWKV_EMPTY_CACHE_EVERY=0
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/ec_test_ws.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
