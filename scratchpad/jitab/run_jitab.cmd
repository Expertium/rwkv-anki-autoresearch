@echo off
REM Direction #2: JIT A/B on the grafted q72u QAT paths. Two identical ~65-step quant-aware runs
REM (full q72u env incl. learnable cbs) on train_db_5k_h1: A = RWKV_NO_JIT=1, B = JIT on.
REM Verdict inputs: exit codes, per-step traces (jitab_nojit.jsonl vs jitab_jit.jsonl), wall times.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\jitab\jitab.log
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

echo ===== JITAB START %DATE% %TIME% ===== > "%LOG%"

echo === A: NO_JIT %TIME% === >> "%LOG%"
set RWKV_NO_JIT=1
set RWKV_STEP_TRACE=scratchpad/jitab/jitab_nojit.jsonl
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/jitab/jitab_ws.toml >> "%LOG%" 2>&1
echo [A_NOJIT_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"
del /Q scratchpad\jitab\jitab_*.pth 2>nul

echo === B: JIT ON %TIME% === >> "%LOG%"
set RWKV_NO_JIT=
set RWKV_STEP_TRACE=scratchpad/jitab/jitab_jit.jsonl
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/jitab/jitab_ws.toml >> "%LOG%" 2>&1
echo [B_JIT_EXIT_%ERRORLEVEL%] %DATE% %TIME% >> "%LOG%"
del /Q scratchpad\jitab\jitab_*.pth 2>nul

echo JITAB_DONE %DATE% %TIME% >> "%LOG%"
