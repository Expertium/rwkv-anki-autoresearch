@echo off
REM ============================================================================
REM BUDGET EXPERIMENT champ5k_b1: champ5k_r1's exact recipe at HALF budget --
REM WS 1 ep (6554 steps) + decay 0.25 ep, quant-aware q72u + learnable cbs,
REM FULL sharded eval 5001-10000, then paired comparison vs champ5k_r1.
REM NO promotion -- this is an A/B on the training budget, recorded manually.
REM No prune env (must run to completion). Launch DETACHED via detach.ps1.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM RWKV_QAT_COMPILE needs MSVC cl.exe on PATH or inductor fails into hollow skipped-batch steps
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\champ5k_b1\champ5k_b1.log
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
set RWKV_STEP_TRACE=scratchpad/champ5k_b1/champ5k_b1_ws_trace.jsonl

echo ===== CHAMP5K_B1 START %DATE% %TIME% ===== > "%LOG%"
echo === WS 1 epoch (1-5000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/champ5k_b1/champ5k_b1_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)

echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/champ5k_b1 champ5kb1ws scratchpad/champ5k_b1/cb_wkv_ws.txt scratchpad/champ5k_b1/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/champ5k_b1/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/champ5k_b1/cb_shift_ws.txt
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.25 ep = ratio 0.25 of 1 WS ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/champ5k_b1 champ5kb1ws champ5kb1d scratchpad/champ5k_b1/champ5k_b1_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/champ5k_b1/champ5k_b1_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/champ5k_b1 champ5kb1d scratchpad/champ5k_b1/cb_wkv_final.txt scratchpad/champ5k_b1/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/champ5k_b1/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/champ5k_b1/cb_shift_final.txt

del /Q result\RWKV-champ5k_b1.jsonl result\RWKV-P-champ5k_b1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/champ5k_b1 champ5kb1d scratchpad/champ5k_b1/champ5k_b1_eval.toml RWKV-champ5k_b1 RWKV-P-champ5k_b1 5001 10000 >> "%LOG%" 2>&1
echo === SHARDED EVAL 5001-10000 (quant-aware, 2 shards) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/champ5k_b1/champ5k_b1_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)

echo === BUDGET A/B: paired vs champ5k_r1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-champ5k_b1.jsonl --cand-imm result/RWKV-P-champ5k_b1.jsonl --champ-ahead result/RWKV-champ5k_r1.jsonl --champ-imm result/RWKV-P-champ5k_r1.jsonl >> "%LOG%" 2>&1
echo BUDGET_AB_DONE (paired_pvalue exit %ERRORLEVEL% is informational, not a failure) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
