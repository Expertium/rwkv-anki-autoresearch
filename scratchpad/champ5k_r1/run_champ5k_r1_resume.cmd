@echo off
REM ============================================================================
REM champ5k_r1 RESUME from the WS->decay seam. WS (13,108 steps) completed and its
REM codebooks resolved; the original decay launch died on the 7-vs-5 optimizer
REM param-group mismatch (fixed in train_rwkv.py 2026-07-08: LEARN=1 resumes now
REM register cb groups BEFORE optimizer.load_state_dict). This .cmd re-runs
REM DECAY -> resolve decay cbs -> sharded FULL eval 5001-10000 -> finish/promote.
REM Appends to the original log. Launch DETACHED via scratchpad/detach.ps1.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM RWKV_QAT_COMPILE needs MSVC cl.exe on PATH or inductor fails into hollow skipped-batch steps
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
set RWKV_QAT_PQ=scratchpad/champ5k_r1/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/champ5k_r1/cb_shift_ws.txt
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

echo ===== CHAMP5K_R1 RESUME (decay onward) %DATE% %TIME% ===== >> "%LOG%"
echo === DECAY (resume) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/champ5k_r1/champ5k_r1_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/champ5k_r1 champ5kd scratchpad/champ5k_r1/cb_wkv_final.txt scratchpad/champ5k_r1/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/champ5k_r1/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/champ5k_r1/cb_shift_final.txt

del /Q result\RWKV-champ5k_r1.jsonl result\RWKV-P-champ5k_r1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/champ5k_r1 champ5kd scratchpad/champ5k_r1/champ5k_r1_eval.toml RWKV-champ5k_r1 RWKV-P-champ5k_r1 5001 10000 >> "%LOG%" 2>&1
echo === SHARDED EVAL 5001-10000 (quant-aware, 2 shards) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/champ5k_r1/champ5k_r1_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)

echo === FINISH: verify n=5000 both modes + PROMOTE %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/champ5k_finish.py champ5k_r1 scratchpad/champ5k_r1/champ5k_r1_ws_trace.jsonl result/RWKV-champ5k_r1.jsonl result/RWKV-P-champ5k_r1.jsonl scratchpad/champ5k_r1 champ5kd scratchpad/champ5k_r1/cb_wkv_final.txt scratchpad/champ5k_r1/cb_shift_final.txt 5000 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
