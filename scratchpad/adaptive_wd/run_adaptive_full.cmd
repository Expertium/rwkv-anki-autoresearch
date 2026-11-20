@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
set RWKV_VRAM_CAP_GB=11
.venv\Scripts\python.exe -u scratchpad\adaptive_wd\adaptive_train.py --run-dir scratchpad\adaptive_wd\adaptive_fullrun_20261104 --start-model pretrain\rwkv\opt_qat46\rwkv_iter46_248.pth --start-optim pretrain\rwkv\opt_qat46\rwkv_iter46_optim_248.pth --max-attempts 30 --qat-env-json scratchpad\adaptive_wd\qat_env.json --full-eval > scratchpad\adaptive_wd\adaptive_fullrun_20261104\run.log 2>&1
set RC=%ERRORLEVEL%
(echo ADAPTIVE_EXIT_%RC%)>> scratchpad\adaptive_wd\adaptive_fullrun_20261104\run.log
exit /b %RC%
