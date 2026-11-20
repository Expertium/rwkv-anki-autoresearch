@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_ID_FEATURES=0
set RWKV_REAL_CYCLES=0
set RWKV_FINAL_LAYOUT=0
set RWKV_E2S_PUBLISHED=0
set RWKV_EXTRA_FEATURES=
set RWKV_ZERO_FEATURES=
set RWKV_AUGMENT_SEED=
.venv\Scripts\python.exe -u -m rwkv.data_processing --config scratchpad\adaptive_wd\build_val_92.toml > scratchpad\adaptive_wd\build_val_92.log 2>&1
set RC=%ERRORLEVEL%
(echo BUILD_VAL_92_EXIT_%RC%)>> scratchpad\adaptive_wd\build_val_92.log
exit /b %RC%
