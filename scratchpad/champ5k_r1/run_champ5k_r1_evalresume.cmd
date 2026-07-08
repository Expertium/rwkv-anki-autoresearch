@echo off
REM ============================================================================
REM champ5k_r1 EVAL RESUME: shard 0 of the first eval died at user 2007/2500 on the
REM per-user-lmdb.open leak (fixed in get_result.py 2026-07-08); shard 1 completed.
REM Shard outputs are resumable (done users are skipped), so this re-runs the sharded
REM eval (only ~494 users compute), re-merges, then verifies n=5000 + promotes.
REM Partial canonical merges are deleted first (merge refuses to clobber).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM identical env to the original eval section (shard-file resume must be numerically
REM consistent with the already-completed user records)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\champ5k_r1\champ5k_r1.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=scratchpad/champ5k_r1/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/champ5k_r1/cb_shift_final.txt
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
set RWKV_STEP_TRACE=

del /Q result\RWKV-champ5k_r1.jsonl result\RWKV-P-champ5k_r1.jsonl 2>nul
echo ===== CHAMP5K_R1 EVAL RESUME %DATE% %TIME% ===== >> "%LOG%"
echo === SHARDED EVAL 5001-10000 (resume; shard files kept) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/champ5k_r1/champ5k_r1_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)

echo === FINISH: verify n=5000 both modes + PROMOTE %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/champ5k_finish.py champ5k_r1 scratchpad/champ5k_r1/champ5k_r1_ws_trace.jsonl result/RWKV-champ5k_r1.jsonl result/RWKV-P-champ5k_r1.jsonl scratchpad/champ5k_r1 champ5kd scratchpad/champ5k_r1/cb_wkv_final.txt scratchpad/champ5k_r1/cb_shift_final.txt 5000 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
