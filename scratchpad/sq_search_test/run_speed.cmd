@echo off
REM Speed A/B for RWKV_SHIFT_SQ_SEARCH=1: same 65+327-step protocol as this morning's flagsC
REM (SQ=0 baseline: steady state (T327-T65)/262 = 1.207 s/step). Same env, same tomls.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sq_search_test\speed.log
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
set RWKV_SHIFT_SQ_SEARCH=1

echo ===== SQ1 SPEED START %DATE% %TIME% ===== > "%LOG%"

echo === SQ1_65 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/jitab/jitab_ws65.toml >> "%LOG%" 2>&1
echo [SQ1_65_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"
del /Q scratchpad\jitab\jitab_*.pth 2>nul

echo === SQ1_327 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/jitab/jitab_ws.toml >> "%LOG%" 2>&1
echo [SQ1_327_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"
del /Q scratchpad\jitab\jitab_*.pth 2>nul

echo SPEED_DONE %DATE% %TIME% >> "%LOG%"
