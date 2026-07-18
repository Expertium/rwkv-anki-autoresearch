@echo off
REM State-norm/NaN probe for A3's eval instability. Args: %1 = user_id, %2 = bf16|fp32.
REM RWKV_NO_JIT=1 is REQUIRED: forward hooks do not fire on ScriptModules.
REM Run only when the GPU is free (d=128 batch forward ~9 GB).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_ARCH_MODULE=scratchpad/track2_a1/architecture_d128_cmix1.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_NO_JIT=1
.venv\Scripts\python.exe -u scratchpad/statenorm/probe_a3_nan.py %1 %2
