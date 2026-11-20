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
.venv\Scripts\python.exe -u scratchpad\adaptive_wd\adaptive_train.py --run-dir scratchpad\adaptive_wd\adaptive_fullrun_92_20261104 --start-model pretrain\rwkv\opt_qat46\rwkv_iter46_248.pth --start-optim pretrain\rwkv\opt_qat46\rwkv_iter46_optim_248.pth --train-db F:/rwkv_lmdb/train_db_5k_h1_adaptive92 --val-db F:/rwkv_lmdb/test_db_5k_adaptive92 --label-db F:/rwkv_lmdb/label_filter_db_id_e2s --max-attempts 30 --qat-env-json scratchpad\adaptive_wd\qat_env.json --full-eval > scratchpad\adaptive_wd\adaptive_fullrun_92_20261104\run.log 2>&1
set RC=%ERRORLEVEL%
(echo ADAPTIVE_EXIT_%RC%)>> scratchpad\adaptive_wd\adaptive_fullrun_92_20261104\run.log
exit /b %RC%
