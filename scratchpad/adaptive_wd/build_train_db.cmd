@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
.venv\Scripts\python.exe -u -m rwkv.data_processing --config scratchpad\adaptive_wd\build_train_db.toml > scratchpad\adaptive_wd\build_train_db.log 2>&1
set RC=%ERRORLEVEL%
(echo BUILD_EXIT_%RC%)>> scratchpad\adaptive_wd\build_train_db.log
exit /b %RC%
