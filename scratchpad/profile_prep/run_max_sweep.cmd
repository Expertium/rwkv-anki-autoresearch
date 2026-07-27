@echo off
REM MAX_TRAIN_GLOBAL_LEN sweep at d=80. MAX was never swept for this trunk -- 32768 was inherited
REM from A18 for step-pairing, and the 110000 sweep was done at d=32 (peak 9.44 GB). At d=80 every
REM arm of the speed benchmark peaked at 12.83 GB on a 12 GB card, i.e. in WDDM paging, which is
REM the likeliest source of the 5.5%% run-to-run spread AND may be costing throughput outright.
REM METRIC = reviews_per_sec, NOT steps_per_sec: changing MAX changes both step count and rows per
REM step, so steps/s is not comparable across arms and reviews/s is.
REM 16384 is the FLOOR -- get_groups silently DROPS any chunk larger than MAX and the largest
REM chunk in train_db_5k_h1 is exactly 16,384 rows.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=scratchpad\profile_prep\max_sweep_hi2.log

set PYTHONUNBUFFERED=1
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
set RWKV_MUON_BATCHED=1
set RWKV_AUGMENT_SEED=1234
set RWKV_DETERMINISTIC=1
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set OMP_NUM_THREADS=7
set RWKV_MAX_STEPS=70
set RWKV_BENCH_WARMUP=30

echo ===== MAX SWEEP START %DATE% %TIME% ===== > "%LOG%"

for %%R in (1 2) do (
  for %%M in (49152 65536 81920) do (
    echo === round %%R MAX %%M === >> "%LOG%"
    .venv\Scripts\python.exe scratchpad/profile_prep/write_max_toml.py %%M >> "%LOG%" 2>&1
    .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/max_sweep_ws.toml 2>&1 | findstr /B /C:"BENCH_RESULT" /C:"Number of groups" >> "%LOG%"
  )
)

echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
