@echo off
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
call scratchpad\adaptive_wd\run_build_train_92.cmd
if errorlevel 1 exit /b %ERRORLEVEL%
call scratchpad\adaptive_wd\run_build_val_92.cmd
if errorlevel 1 exit /b %ERRORLEVEL%
exit /b 0
