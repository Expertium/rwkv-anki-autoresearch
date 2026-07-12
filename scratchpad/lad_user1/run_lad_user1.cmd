@echo off
REM ============================================================================
REM STATE-SIZE LADDER, USER/GLOBAL RUNG 1 lad_user1 (iter 6): champion recipe +
REM user n_heads 2->1 (RWKV_STREAM_HEADS=user:1) -- user per-entity state 1.89x
REM at -198 params, card/note/deck/preset UNCHANGED. WS 1 ep (6554) + decay
REM 0.25 ep, quant-aware q72u + learnable cbs, FULL eval 5001-10000 with
REM SEQUENTIAL shards (lesson from iter 5: user-K=32 has the same 1M-token
REM chunk-state VRAM surcharge as preset-K=32 -- 2 parallel shards WDDM-thrash),
REM paired vs champ5k_b1 (iter 2). CANDIDATE run: VALIDATION PRUNE ON.
REM Gate + promotion MANUAL. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM RWKV_QAT_COMPILE needs MSVC cl.exe on PATH or inductor fails into hollow skipped-batch steps
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\lad_user1\lad_user1.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_STREAM_HEADS=user:1
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
set RWKV_STEP_TRACE=scratchpad/lad_user1/lad_user1_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k.json

echo ===== LAD_USER1 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\lad_user1\lad_user1_ws_trace.jsonl scratchpad\lad_user1\lad_user1_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, user H=1, vprune on) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/lad_user1/lad_user1_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)

echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/lad_user1 laduser1ws scratchpad/lad_user1/cb_wkv_ws.txt scratchpad/lad_user1/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/lad_user1/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/lad_user1/cb_shift_ws.txt
set RWKV_STEP_TRACE=
set RWKV_VPRUNE_REF=

echo === DECAY SETUP (0.25 ep = ratio 0.25 of 1 WS ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/lad_user1 laduser1ws laduser1d scratchpad/lad_user1/lad_user1_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/lad_user1/lad_user1_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/lad_user1 laduser1d scratchpad/lad_user1/cb_wkv_final.txt scratchpad/lad_user1/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/lad_user1/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/lad_user1/cb_shift_final.txt

del /Q result\RWKV-lad_user1.jsonl result\RWKV-P-lad_user1.jsonl result\RWKV-lad_user1-s0.jsonl result\RWKV-P-lad_user1-s0.jsonl result\RWKV-lad_user1-s1.jsonl result\RWKV-P-lad_user1-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/lad_user1 laduser1d scratchpad/lad_user1/lad_user1_eval.toml RWKV-lad_user1 RWKV-P-lad_user1 5001 10000 >> "%LOG%" 2>&1
echo === EVAL PLAN (dry-run writes shard tomls + user lists) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/lad_user1/lad_user1_eval.toml --dry-run >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_PLANFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
echo === SHARD 0 SEQUENTIAL %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/eval_shards/shard_0.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_S0FAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo === SHARD 1 SEQUENTIAL %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/eval_shards/shard_1.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_S1FAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 8
)
echo === MERGE (eval_sharded relaunch: shards skip all users, then merge) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/lad_user1/lad_user1_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_MERGEFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)

echo === GATE: paired vs champ5k_b1 (iter 2) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-lad_user1.jsonl --cand-imm result/RWKV-P-lad_user1.jsonl --champ-ahead result/RWKV-champ5k_b1.jsonl --champ-imm result/RWKV-P-champ5k_b1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%: 0 = both p-gates pass) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
