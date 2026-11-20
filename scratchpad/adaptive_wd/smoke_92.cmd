@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_ID_FEATURES=0
set RWKV_REAL_CYCLES=0
set RWKV_VRAM_CAP_GB=11
.venv\Scripts\python.exe -u scratchpad\adaptive_wd\adaptive_train.py --smoke --smoke-from-scratch --run-dir scratchpad\adaptive_wd\smoke_92_20261105 --train-db F:/rwkv_lmdb/train_db_5k_h1_adaptive92 --val-db F:/rwkv_lmdb/test_db_5k_adaptive92 --label-db F:/rwkv_lmdb/label_filter_db_id_e2s > scratchpad\adaptive_wd\smoke_92_20261105\smoke.log 2>&1
set RC=%ERRORLEVEL%
(echo SMOKE_EXIT_%RC%)>> scratchpad\adaptive_wd\smoke_92_20261105\smoke.log
exit /b %RC%
