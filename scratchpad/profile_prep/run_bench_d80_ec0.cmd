@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=scratchpad\profile_prep\bench_d80_ec0.log

set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_GRU_HEAD=3
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_AUGMENT_SEED=1234
set RWKV_DETERMINISTIC=1
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_EMPTY_CACHE_WINDOW=0
set OMP_NUM_THREADS=7

set RWKV_MAX_STEPS=90
set RWKV_BENCH_WARMUP=40

echo ===== PROFILE d=80 trunk START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/profile_d80_ws.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
