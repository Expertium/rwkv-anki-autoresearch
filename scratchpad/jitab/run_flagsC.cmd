@echo off
REM Variant C: the sibling's SANCTIONED round-4 production flag set (COMPILE=student + ROT_CACHE +
REM FAST_EMB + EMA_FOREACH + NO_MEMFILL) under NO_JIT, on the full q72u 5k env. Two lengths (65 +
REM 327 steps) so startup and steady-state separate exactly, comparable to A (NO_JIT) and B (JIT).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM torch.compile (inductor) needs MSVC cl.exe on PATH -- without vcvars every compile attempt
REM dies as "Compiler: cl is not found", the NaN-except swallows it, and steps go hollow.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\jitab\flagsC.log
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt
set RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_q72u.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=student
set RWKV_QAT_ROT_CACHE=1
set RWKV_QAT_FAST_EMB=1
set RWKV_QAT_EMA_FOREACH=1
set RWKV_QAT_NO_MEMFILL=1

echo ===== FLAGSC START %DATE% %TIME% ===== > "%LOG%"

echo === C65 %TIME% === >> "%LOG%"
set RWKV_STEP_TRACE=scratchpad/jitab/jitab_c65.jsonl
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/jitab/jitab_ws65.toml >> "%LOG%" 2>&1
echo [C65_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"
del /Q scratchpad\jitab\jitab_*.pth 2>nul

echo === C327 %TIME% === >> "%LOG%"
set RWKV_STEP_TRACE=scratchpad/jitab/jitab_c327.jsonl
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/jitab/jitab_ws.toml >> "%LOG%" 2>&1
echo [C327_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"
del /Q scratchpad\jitab\jitab_*.pth 2>nul

echo FLAGSC_DONE %DATE% %TIME% >> "%LOG%"
