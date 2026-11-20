@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
.venv\Scripts\python.exe -m rwkv.data_processing --config scratchpad\adaptive_wd\build_train_db.toml --help > scratchpad\adaptive_wd\build_help.log 2>&1
set RC=%ERRORLEVEL%
(echo HELP_EXIT_%RC%)>> scratchpad\adaptive_wd\build_help.log
exit /b %RC%
