@echo off
REM fp32 NaN probe (iter19): re-eval user 8902 (nan-skipped in bf16) at DTYPE=float.
REM NaN persists in fp32 -> weight-level (A0-class); finite -> bf16-transient.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter19_pbin025\probe32.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_ZERO_FEATURES=22
set RWKV_EVAL_CAST_FP32=1
echo ===== I19 FP32 PROBE (user 8902) START %DATE% %TIME% ===== > "%LOG%"
del /Q result\RWKV-i19probe32.jsonl result\RWKV-P-i19probe32.jsonl 2>nul
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/iter19_pbin025/probe_fp32.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
