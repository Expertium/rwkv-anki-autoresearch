@echo off
REM ============================================================================
REM RESEARCH PHASE iter 10: iter10_kd = WARMUP-ONLY KD from the d=128 teacher.
REM For the first 800 WS steps the loss targets are alpha*teacher + (1-alpha)*
REM hard labels, alpha linear 1 -> 0 (RWKV_KD_MIX -> stored dump from
REM run_kd_dump.cmd; per-step checksum asserts batch-stream identity, mismatch
REM = exit 43). Everything else = EXACT champ5k_b1 recipe: champion arch
REM (H=2/K=16), champion HPs, seed 1234, WS 1 ep (6554) + decay 0.25 ep,
REM quant-aware q72u + learnable cbs, FULL eval 5001-10000 (PARALLEL 2 shards),
REM paired vs champ5k_b1 (iter 2). Vprune ON standard deltas (val = hard labels
REM always; window 800 < vprune min_step 1000). Tested SEPARATELY from iter 9's
REM init change per the interaction warning. Gate + promotion MANUAL.
REM PREREQ: full 800-step dump present in scratchpad/iter10_kd/dump.
REM Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM RWKV_QAT_COMPILE needs MSVC cl.exe on PATH or inductor fails into hollow skipped-batch steps
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter10_kd\iter10_kd.log
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
set RWKV_KD_MIX=scratchpad/iter10_kd/dump:800
set RWKV_STEP_TRACE=scratchpad/iter10_kd/iter10_kd_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006

echo ===== ITER10_KD START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\iter10_kd\iter10_kd_ws_trace.jsonl scratchpad\iter10_kd\iter10_kd_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, warmup-KD window 800, vprune on) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter10_kd/iter10_kd_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if %ERRORLEVEL%==43 (
  echo DONE_EXIT_KDMISMATCH %DATE% %TIME% >> "%LOG%"
  exit /b 43
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)

REM KD is a WS-warmup-only lever; the decay phase re-seeds random and REPRODUCES
REM the epoch-0 batch stream, so the checksum canNOT catch a misfire -- clear it.
set RWKV_KD_MIX=

echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/iter10_kd iter10kdws scratchpad/iter10_kd/cb_wkv_ws.txt scratchpad/iter10_kd/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/iter10_kd/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/iter10_kd/cb_shift_ws.txt
set RWKV_STEP_TRACE=
set RWKV_VPRUNE_REF=

echo === DECAY SETUP (0.25 ep = ratio 0.25 of 1 WS ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter10_kd iter10kdws iter10kdd scratchpad/iter10_kd/iter10_kd_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter10_kd/iter10_kd_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/iter10_kd iter10kdd scratchpad/iter10_kd/cb_wkv_final.txt scratchpad/iter10_kd/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/iter10_kd/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/iter10_kd/cb_shift_final.txt

del /Q result\RWKV-iter10_kd.jsonl result\RWKV-P-iter10_kd.jsonl result\RWKV-iter10_kd-s0.jsonl result\RWKV-P-iter10_kd-s0.jsonl result\RWKV-iter10_kd-s1.jsonl result\RWKV-P-iter10_kd-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter10_kd iter10kdd scratchpad/iter10_kd/iter10_kd_eval.toml RWKV-iter10_kd RWKV-P-iter10_kd 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (2 PARALLEL shards + merge; champion arch, no elevated VRAM) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter10_kd/iter10_kd_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs champ5k_b1 (iter 2) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter10_kd.jsonl --cand-imm result/RWKV-P-iter10_kd.jsonl --champ-ahead result/RWKV-champ5k_b1.jsonl --champ-imm result/RWKV-P-champ5k_b1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%: 0 = both p-gates pass) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
