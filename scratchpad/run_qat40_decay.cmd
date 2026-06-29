@echo off
REM Detached decay+QAT (iter40). Log goes to a STABLE repo path (session temp dirs rotate on Esc).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_NO_JIT=1
set RWKV_QAT_SCOPE=card:int2,note:int4
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat40_decay.log
echo START %DATE% %TIME% > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config rwkv/train_rwkv_config_iter40_qat_decay.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
