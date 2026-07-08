@echo off
REM ============================================================================
REM FIRST REAL 5k CHAMPION RUN (champ5k_r1): WS 2 ep (13,108 steps) -> resolve cbs ->
REM cosine decay 0.5 ep (ratio 0.25, champion HPs) -> resolve cbs -> FULL quant-aware
REM sharded eval 5001-10000 -> promote to optimization/champion_5k.json.
REM Env = the FROZEN 5k-family env (f544a31) + per-run learnable codebooks (a3fac9f).
REM No prune env: this run CREATES the Wilcoxon prune reference.
REM Launch DETACHED via scratchpad/detach.ps1 (survives Esc/session teardown).
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
set RWKV_STEP_TRACE=scratchpad/champ5k_r1/champ5k_r1_ws_trace.jsonl

echo ===== CHAMP5K_R1 START %DATE% %TIME% ===== > "%LOG%"
echo === WS 2 epochs (1-5000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/champ5k_r1/champ5k_r1_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)

echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/champ5k_r1 champ5kws scratchpad/champ5k_r1/cb_wkv_ws.txt scratchpad/champ5k_r1/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/champ5k_r1/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/champ5k_r1/cb_shift_ws.txt
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.5 ep = ratio 0.25) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/champ5k_r1 champ5kws champ5kd scratchpad/champ5k_r1/champ5k_r1_decay.toml train_db_5k_h1 1 5000 0.5 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
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
