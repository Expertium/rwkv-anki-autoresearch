@echo off
REM TEACHER-RETRAIN COST MEASUREMENT (2026-09-04): a d=128 model (scratchpad/architecture_old_d128.py)
REM on the gen-5 SSD dbs with the 109-dim layout and the modern recipe minus the two d80-specific
REM strip flags, for 0.01 epoch at MAX 32768 and then at MAX 65536. Reports steps/s per config and
REM whether 65536 fits the 12 GB card. Feeds Andrew's budget decision for PROPOSALS order 7; nothing
REM here is a research iteration and no checkpoint is kept for use.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\teacher_gen5
set LOG=%DIR%\t128_timing.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_MUON_INCLUDE_LORA=1
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
set RWKV_ZERO_FEATURES=
set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_id5
set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_id5
set RWKV_LABEL_FILTER_DB=F:/rwkv_lmdb/label_filter_db_id_e2s
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_STRIP_CMIX=
set RWKV_STRIP_L0_VLORA=

echo ===== T128 TIMING START %DATE% %TIME% ===== > "%LOG%"

echo === MAX 32768 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/teacher_gen5/ws_t128_m32k.toml > "%DIR%\ws_m32k_%STAMP%.log" 2>&1
echo m32k rc %ERRORLEVEL% %TIME% >> "%LOG%"
if not exist "%DIR%\t128_m32k_50.pth" (
  echo m32k produced no step-50 checkpoint -- the phase did not run %TIME% >> "%LOG%"
)
findstr /C:"Trainable parameters" "%DIR%\ws_m32k_%STAMP%.log" >> "%LOG%"
findstr /C:"Steps per second" "%DIR%\ws_m32k_%STAMP%.log" >> "%LOG%"
findstr /I /C:"OutOfMemory" /C:"Traceback" "%DIR%\ws_m32k_%STAMP%.log" >> "%LOG%"

echo === MAX 65536 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/teacher_gen5/ws_t128_m64k.toml > "%DIR%\ws_m64k_%STAMP%.log" 2>&1
echo m64k rc %ERRORLEVEL% %TIME% >> "%LOG%"
if not exist "%DIR%\t128_m64k_50.pth" (
  echo m64k produced no step-50 checkpoint -- the phase did not run or OOM'd %TIME% >> "%LOG%"
)
findstr /C:"Trainable parameters" "%DIR%\ws_m64k_%STAMP%.log" >> "%LOG%"
findstr /C:"Steps per second" "%DIR%\ws_m64k_%STAMP%.log" >> "%LOG%"
findstr /I /C:"OutOfMemory" /C:"Traceback" "%DIR%\ws_m64k_%STAMP%.log" >> "%LOG%"

echo T128_TIMING_OK %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
