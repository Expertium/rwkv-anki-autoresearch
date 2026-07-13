@echo off
REM ============================================================================
REM RESEARCH PHASE iter 11: iter11_gemb = additive grade embedding (RWKV_GRADE_
REM EMB=1): x = features2card(f) + grade_onehot @ E, E 4x32 zero-init -- a
REM dedicated bypass around the shared input MLP for the grade signal. +128
REM params (193,852 <= 225k). Everything else = EXACT champ5k_b1 recipe:
REM champion arch (H=2/K=16), champion HPs, seed 1234, WS 1 ep (6554) + decay
REM 0.25 ep, quant-aware q72u + learnable cbs, FULL eval 5001-10000 (PARALLEL
REM 2 shards -- champion arch, no elevated VRAM), paired vs champ5k_b1 (iter 2).
REM CANDIDATE run: VALIDATION PRUNE ON standard deltas (matched regularization,
REM same seed). Gate + promotion MANUAL. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM RWKV_QAT_COMPILE needs MSVC cl.exe on PATH or inductor fails into hollow skipped-batch steps
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter11_gemb\iter11_gemb.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_GRADE_EMB=1
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
set RWKV_STEP_TRACE=scratchpad/iter11_gemb/iter11_gemb_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER11_GEMB START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter11_gemb\iter11_gemb_ws_trace.jsonl scratchpad\iter11_gemb\iter11_gemb_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, grade emb on, vprune on) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter11_gemb/iter11_gemb_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)

echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/iter11_gemb iter11gembws scratchpad/iter11_gemb/cb_wkv_ws.txt scratchpad/iter11_gemb/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/iter11_gemb/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/iter11_gemb/cb_shift_ws.txt
set RWKV_STEP_TRACE=
set RWKV_VPRUNE_REF=

echo === DECAY SETUP (0.25 ep = ratio 0.25 of 1 WS ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter11_gemb iter11gembws iter11gembd scratchpad/iter11_gemb/iter11_gemb_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter11_gemb/iter11_gemb_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/iter11_gemb iter11gembd scratchpad/iter11_gemb/cb_wkv_final.txt scratchpad/iter11_gemb/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/iter11_gemb/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/iter11_gemb/cb_shift_final.txt

del /Q result\RWKV-iter11_gemb.jsonl result\RWKV-P-iter11_gemb.jsonl result\RWKV-iter11_gemb-s0.jsonl result\RWKV-P-iter11_gemb-s0.jsonl result\RWKV-iter11_gemb-s1.jsonl result\RWKV-P-iter11_gemb-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter11_gemb iter11gembd scratchpad/iter11_gemb/iter11_gemb_eval.toml RWKV-iter11_gemb RWKV-P-iter11_gemb 5001 10000 >> "%LOG%" 2>&1
REM SEQUENTIAL shards for ALL evals (2026-07-13: the iter-10 parallel eval wedged
REM on the CHAMPION arch -- two mega-users collided into WDDM oversubscription;
REM the iter-5 "elevated-VRAM only" rule was too narrow. Sequential is ~45 min
REM slower when parallel would have worked but never wedges = unattended-safe.)
echo === EVAL PLAN (dry-run writes shard tomls + user lists) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter11_gemb/iter11_gemb_eval.toml --dry-run >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter11_gemb/iter11_gemb_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_MERGEFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)

echo === GATE: paired vs champ5k_b1 (iter 2) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter11_gemb.jsonl --cand-imm result/RWKV-P-iter11_gemb.jsonl --champ-ahead result/RWKV-champ5k_b1.jsonl --champ-imm result/RWKV-P-champ5k_b1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%: 0 = both p-gates pass) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
