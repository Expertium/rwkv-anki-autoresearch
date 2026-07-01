@echo off
REM Build the sc8k train_db for users 1000-2499 (1500 users) for the "varied data, few epochs" experiment.
REM CPU-only (DEVICE=cpu in the config), resumable (skips _done users). ~56 GB, est ~1-2 hr. Detached.
REM Monitor scratchpad/build_1500.log (tqdm "Generating Data" progress; DONE_EXIT_ at end).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\build_1500.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
echo ===== BUILD train_db_sc8k_1500 (users 1000-2499) START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.data_processing --config rwkv/data_processing_config_sc8k_1500.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
