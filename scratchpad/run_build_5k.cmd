@echo off
REM 5k-phase LMDB build, 1 thread (Andrew 2026-07-01). Runs the 6 builds sequentially; each is resumable
REM (skips done users) so re-launching after any interruption continues. Order front-loads the 5001-10000
REM eval data (steps 1-2) so the d=128 baseline eval can run while the big train_db(1-5000) builds (step 3).
REM Detached via detach.ps1 (survives Esc). Monitor scratchpad/build_5k.log + FREE space on C:/F: (OS truth).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PY=.venv\Scripts\python.exe
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\build_5k.log
set PYTHONUNBUFFERED=1
echo ===== BUILD 5K START %DATE% %TIME% ===== >> "%LOG%"

echo [STEP1 find_equalize 5001-10000] %DATE% %TIME% >> "%LOG%"
%PY% -u -m rwkv.find_equalize_test_reviews --config rwkv/find_equalize_5k_h2.toml >> "%LOG%" 2>&1
echo [STEP1_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"

echo [STEP2 test_db 5001-10000 F:] %DATE% %TIME% >> "%LOG%"
%PY% -u -m rwkv.data_processing --config rwkv/data_processing_test_5k_h2.toml >> "%LOG%" 2>&1
echo [STEP2_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"

echo [STEP3 train_db 1-5000 C:] %DATE% %TIME% >> "%LOG%"
%PY% -u -m rwkv.data_processing --config rwkv/data_processing_train_5k_h1.toml >> "%LOG%" 2>&1
echo [STEP3_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"

echo [STEP4 find_equalize 1-5000] %DATE% %TIME% >> "%LOG%"
%PY% -u -m rwkv.find_equalize_test_reviews --config rwkv/find_equalize_5k_h1.toml >> "%LOG%" 2>&1
echo [STEP4_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"

echo [STEP5 test_db 1-5000 F:] %DATE% %TIME% >> "%LOG%"
%PY% -u -m rwkv.data_processing --config rwkv/data_processing_test_5k_h1.toml >> "%LOG%" 2>&1
echo [STEP5_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"

echo [STEP6 train_db 5001-10000 F:] %DATE% %TIME% >> "%LOG%"
%PY% -u -m rwkv.data_processing --config rwkv/data_processing_train_5k_h2.toml >> "%LOG%" 2>&1
echo [STEP6_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"

echo ===== BUILD 5K ALLDONE %DATE% %TIME% ===== >> "%LOG%"
