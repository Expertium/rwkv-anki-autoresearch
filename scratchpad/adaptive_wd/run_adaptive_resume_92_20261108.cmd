@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_ID_FEATURES=0
set RWKV_REAL_CYCLES=0
set RWKV_FINAL_LAYOUT=0
set RWKV_E2S_PUBLISHED=0
set RWKV_EXTRA_FEATURES=
set RWKV_ZERO_FEATURES=
set RWKV_VRAM_CAP_GB=11
.venv\Scripts\python.exe -u scratchpad\adaptive_wd\adaptive_train.py --run-dir scratchpad\adaptive_wd\adaptive_fullrun_92_20261105 --resume-state scratchpad\adaptive_wd\adaptive_fullrun_92_20261105\adaptive_state.json --train-db F:/rwkv_lmdb/train_db_5k_h1_adaptive92 --val-db F:/rwkv_lmdb/test_db_5k_adaptive92 --label-db F:/rwkv_lmdb/label_filter_db_id_e2s --max-failed-epochs 7 --max-successful-epochs 10 --qat-env-json scratchpad\adaptive_wd\qat_env.json --full-eval > scratchpad\adaptive_wd\adaptive_fullrun_92_20261105\resume_controller_20261109.log 2>&1
set RC=%ERRORLEVEL%
(echo ADAPTIVE_RESUME_EXIT_%RC%)>> scratchpad\adaptive_wd\adaptive_fullrun_92_20261105\resume_controller_20261109.log
exit /b %RC%
