@echo off
REM ===========================================================================================
REM V1 MAX sweep: is the 12x slowdown DISPATCH or MEMORY PRESSURE?
REM
REM The compile-on arm measured peak_reserved_gb=12.583 on a 12 GB card -- over the ceiling and
REM into WDDM paging, which CLAUDE.md records as depressing every arm of a measurement. The FSRS
REM T-loop saves ~40 intermediates per review for backward, so its activation memory scales with
REM the batch's review count in a way the fused WKV kernel's did not.
REM
REM Three MAX values, 220 steps each. COMPARE ON reviews_per_sec, never steps_per_sec: changing
REM MAX changes both the numerator and what a step contains.
REM
REM If rev/s RISES as MAX falls, the slowdown is paging and V1 wants a smaller batch.
REM If rev/s is flat or falls, it is dispatch, and no batch size fixes it.
REM
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hybrid100k
set LOG=%DIR%\maxsweep_v1.log
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

echo ===== V1 MAX SWEEP START %DATE% %TIME% ===== >> "%LOG%"
REM compile is OFF: measured 0.1658 vs 0.1537 steps/s on this workload, i.e. it HURTS
REM here (dynamic sequence lengths defeat it), inverting the standard stack.
set RWKV_QAT_COMPILE=

for %%M in (65536 32768 16384) do (
  echo --- MAX=%%M %TIME% >> "%LOG%"
  .venv\Scripts\python.exe scratchpad/write_max_toml.py scratchpad/hybrid100k/hyv1_ws.toml %%M "%DIR%\hyv1_max%%M.toml" >> "%LOG%" 2>&1
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%DIR%\hyv1_max%%M.toml" > "%DIR%\max%%M_%STAMP%.log" 2>&1
  findstr /C:"BENCH_RESULT" "%DIR%\max%%M_%STAMP%.log" >> "%LOG%"
)
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
