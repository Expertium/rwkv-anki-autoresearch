@echo off
REM ============================================================================
REM RESEARCH PHASE iter 9: iter9_sp = SHRINK-PERTURB INIT (lambda=0.5, fresh
REM seed 777) from the champion final ckpt champ5kb1d_1638.pth, via the new
REM RWKV_INIT_BLEND hook. Everything else = EXACT champ5k_b1 recipe: champion
REM arch (H=2/K=16, no stream overrides), champion HPs, seed 1234, WS 1 ep
REM (6554) + decay 0.25 ep, quant-aware q72u + learnable cbs, FULL eval
REM 5001-10000 (PARALLEL 2 shards -- champion arch, no elevated VRAM), paired
REM vs champ5k_b1 (iter 2). CANDIDATE run: VALIDATION PRUNE ON at standard
REM deltas 0.004/0.006 (same seed as the champion ref trace). NOTE: the blend
REM starts training much closer to a solution, so early val should only be
REM BETTER than the ref -- vprune cannot false-kill on that side.
REM Gate + promotion MANUAL. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM RWKV_QAT_COMPILE needs MSVC cl.exe on PATH or inductor fails into hollow skipped-batch steps
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter9_sp\iter9_sp.log
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
set RWKV_INIT_BLEND=scratchpad/champ5k_b1/champ5kb1d_1638.pth:0.5:777
set RWKV_STEP_TRACE=scratchpad/iter9_sp/iter9_sp_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER9_SP START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter9_sp\iter9_sp_ws_trace.jsonl scratchpad\iter9_sp\iter9_sp_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, shrink-perturb lam=0.5 seed 777, vprune on) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter9_sp/iter9_sp_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)

REM The blend is a WS-init-only hook; the decay phase warm-starts from the WS
REM ckpt via LOAD_MODEL and MUST NOT see RWKV_INIT_BLEND (the hook asserts).
set RWKV_INIT_BLEND=

echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/iter9_sp iter9spws scratchpad/iter9_sp/cb_wkv_ws.txt scratchpad/iter9_sp/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/iter9_sp/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/iter9_sp/cb_shift_ws.txt
set RWKV_STEP_TRACE=
set RWKV_VPRUNE_REF=

echo === DECAY SETUP (0.25 ep = ratio 0.25 of 1 WS ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter9_sp iter9spws iter9spd scratchpad/iter9_sp/iter9_sp_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter9_sp/iter9_sp_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/iter9_sp iter9spd scratchpad/iter9_sp/cb_wkv_final.txt scratchpad/iter9_sp/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/iter9_sp/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/iter9_sp/cb_shift_final.txt

del /Q result\RWKV-iter9_sp.jsonl result\RWKV-P-iter9_sp.jsonl result\RWKV-iter9_sp-s0.jsonl result\RWKV-P-iter9_sp-s0.jsonl result\RWKV-iter9_sp-s1.jsonl result\RWKV-P-iter9_sp-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter9_sp iter9spd scratchpad/iter9_sp/iter9_sp_eval.toml RWKV-iter9_sp RWKV-P-iter9_sp 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (2 PARALLEL shards + merge; champion arch, no elevated VRAM) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter9_sp/iter9_sp_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs champ5k_b1 (iter 2) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter9_sp.jsonl --cand-imm result/RWKV-P-iter9_sp.jsonl --champ-ahead result/RWKV-champ5k_b1.jsonl --champ-imm result/RWKV-P-champ5k_b1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%: 0 = both p-gates pass) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
