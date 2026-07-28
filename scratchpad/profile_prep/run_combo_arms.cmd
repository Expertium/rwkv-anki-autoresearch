@echo off
REM Do the three individually-real wins COMPOSE? base vs combo, 4 rounds, round-robin.
REM Individually (clean card, 1.8%% noise floor): muon 1.053x, compile 1.053x, fetch2 1.061x.
REM They attack different things -- muon cuts optimizer dispatches, compile fuses the mixer
REM elementwise soup, fetch2 cuts CPU contention -- so they SHOULD stack, but that is a
REM prediction, not a measurement.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=scratchpad\profile_prep\combo_arms.log

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
set RWKV_AUGMENT_SEED=1234
set RWKV_DETERMINISTIC=1
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set OMP_NUM_THREADS=7
set RWKV_MAX_STEPS=70
set RWKV_BENCH_WARMUP=30

echo ===== COMBO ARMS START %DATE% %TIME% ===== > "%LOG%"

for %%R in (1 2 3 4) do (
  echo === round %%R arm base %TIME% === >> "%LOG%"
  set RWKV_NO_JIT=0
  set RWKV_MUON_BATCHED=0
  set RWKV_QAT_COMPILE=
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/profile_d80_ws.toml 2>&1 | findstr /B /C:"BENCH_RESULT" /C:"[compile]" /C:"[muon]" >> "%LOG%"

  echo === round %%R arm combo %TIME% === >> "%LOG%"
  set RWKV_NO_JIT=1
  set RWKV_MUON_BATCHED=1
  set RWKV_QAT_COMPILE=1
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/profile_d80_fetch2.toml 2>&1 | findstr /B /C:"BENCH_RESULT" /C:"[compile]" /C:"[muon]" >> "%LOG%"
)

echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
