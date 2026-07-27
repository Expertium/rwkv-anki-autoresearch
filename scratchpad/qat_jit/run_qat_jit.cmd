@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=scratchpad\qat_jit\qat_jit.log

REM ---- iter-31 trunk env (identical in both arms) ----
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

REM ---- the frozen q72u quant-aware recipe ----
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt
set RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_q72u.txt
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1

set RWKV_MAX_STEPS=80
set RWKV_BENCH_WARMUP=30

echo ===== QAT-JIT A/B START %DATE% %TIME% ===== > "%LOG%"

echo === ARM A: RWKV_NO_JIT=1 (the current forced path) %TIME% === >> "%LOG%"
del /q scratchpad\qat_jit\trace_nojit.jsonl 2>nul
set RWKV_NO_JIT=1
set RWKV_STEP_TRACE=scratchpad/qat_jit/trace_nojit.jsonl
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/qat_jit/qat_jit_ws.toml >> "%LOG%" 2>&1
if errorlevel 1 (
  echo DONE_EXIT_ARMAFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 1
)

echo === ARM A2: RWKV_NO_JIT=1 AGAIN (NULL CONTROL, same flags as arm A) %TIME% === >> "%LOG%"
del /q scratchpad\qat_jit\trace_nojit2.jsonl 2>nul
set RWKV_NO_JIT=1
set RWKV_STEP_TRACE=scratchpad/qat_jit/trace_nojit2.jsonl
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/qat_jit/qat_jit_ws.toml >> "%LOG%" 2>&1
if errorlevel 1 (
  echo DONE_EXIT_ARMA2FAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 1
)

echo === ARM B: JIT ON %TIME% === >> "%LOG%"
del /q scratchpad\qat_jit\trace_jit.jsonl 2>nul
set RWKV_NO_JIT=0
set RWKV_STEP_TRACE=scratchpad/qat_jit/trace_jit.jsonl
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/qat_jit/qat_jit_ws.toml >> "%LOG%" 2>&1
if errorlevel 1 (
  echo DONE_EXIT_ARMBFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 1
)

echo === COMPARE %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad\qat_jit\compare_traces.py >> "%LOG%" 2>&1

echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
