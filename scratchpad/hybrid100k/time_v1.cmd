@echo off
REM ===========================================================================================
REM Why is V1 11.7x slower than the hybrid arms, and is torch.compile helping or hurting?
REM
REM V1 replaces a FUSED CUDA WKV kernel (one launch per layer) with a Python loop of ~40
REM elementwise ops per review per card sequence. The training step is already 85% CPU-DISPATCH
REM bound (optimization/TRAINING_SPEED.md), so adding thousands of dispatches is the worst
REM possible shape. Measured: 0.123 steps/s at step 260, warmup provably over -- about 50 h for
REM a full run, against 6 h for an arm.
REM
REM RWKV_QAT_COMPILE fuses the mixer forwards. With a dynamic-length Python loop it may be
REM RECOMPILING per sequence-length bucket, in which case it is not merely useless but harmful.
REM Two arms, RWKV_MAX_STEPS=220 each, timed past warmup by the trainer's own bench counter.
REM
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hybrid100k
set LOG=%DIR%\time_v1.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/hybrid100k/arch_fsrs_v1.py
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2
set RWKV_FSRS_CARD=0
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
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_KD_MIX=C:\rwkv_kd_dump\t128_seedpair_65k:10935
set RWKV_KD_ALPHA=0.9
set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120

echo ===== V1 TIMING START %DATE% %TIME% ===== >> "%LOG%"

REM ---- ARM 1: compile ON (what the standard stack sets) ----
set RWKV_QAT_COMPILE=1
echo --- arm compile_on %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/hybrid100k/hyv1_ws.toml > "%DIR%\time_on_%STAMP%.log" 2>&1
echo   exit=%ERRORLEVEL% %TIME% >> "%LOG%"

REM ---- ARM 2: compile OFF ----
set RWKV_QAT_COMPILE=
echo --- arm compile_off %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/hybrid100k/hyv1_ws.toml > "%DIR%\time_off_%STAMP%.log" 2>&1
echo   exit=%ERRORLEVEL% %TIME% >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
