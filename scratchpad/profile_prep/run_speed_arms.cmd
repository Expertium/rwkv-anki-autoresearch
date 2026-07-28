@echo off
REM Multi-arm training-speed A/B, ROUND-ROBIN with repeats.
REM Round-robin (A,B,C,D, A,B,C,D, ...) rather than blocked, so thermal/background drift hits
REM every arm equally; medians over rounds then beat the ~5% run-to-run noise measured on
REM identical flags (scratchpad/qat_jit).
REM   base    = exactly what training does today (JIT on)
REM   muon    = base + RWKV_MUON_BATCHED=1          <- directly adoptable
REM   nojit   = RWKV_NO_JIT=1                        <- compile's required baseline
REM   compile = nojit + RWKV_QAT_COMPILE=1           <- fuses the mixer elementwise soup
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=scratchpad\profile_prep\speed_arms2.log

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

echo ===== SPEED ARMS START %DATE% %TIME% ===== > "%LOG%"

for %%R in (1 2 3) do (
  echo === round %%R arm base %TIME% === >> "%LOG%"
  set RWKV_NO_JIT=0
  set RWKV_MUON_BATCHED=0
  set RWKV_QAT_COMPILE=
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/profile_d80_ws.toml 2>&1 | findstr /B /C:"BENCH_RESULT" /C:"[compile]" /C:"[muon]" /C:"[jit]" >> "%LOG%"

  echo === round %%R arm muon %TIME% === >> "%LOG%"
  set RWKV_NO_JIT=0
  set RWKV_MUON_BATCHED=1
  set RWKV_QAT_COMPILE=
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/profile_d80_ws.toml 2>&1 | findstr /B /C:"BENCH_RESULT" /C:"[compile]" /C:"[muon]" /C:"[jit]" >> "%LOG%"

  echo === round %%R arm nojit %TIME% === >> "%LOG%"
  set RWKV_NO_JIT=1
  set RWKV_MUON_BATCHED=0
  set RWKV_QAT_COMPILE=
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/profile_d80_ws.toml 2>&1 | findstr /B /C:"BENCH_RESULT" /C:"[compile]" /C:"[muon]" /C:"[jit]" >> "%LOG%"

  echo === round %%R arm compile %TIME% === >> "%LOG%"
  set RWKV_NO_JIT=1
  set RWKV_MUON_BATCHED=0
  set RWKV_QAT_COMPILE=1
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/profile_d80_ws.toml 2>&1 | findstr /B /C:"BENCH_RESULT" /C:"[compile]" /C:"[muon]" /C:"[jit]" >> "%LOG%"
  echo === round %%R arm fetch2 %TIME% === >> "%LOG%"
  set RWKV_NO_JIT=0
  set RWKV_MUON_BATCHED=0
  set RWKV_QAT_COMPILE=
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/profile_prep/profile_d80_fetch2.toml 2>&1 | findstr /B /C:"BENCH_RESULT" /C:"[compile]" /C:"[muon]" /C:"[jit]" >> "%LOG%"
)

echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
