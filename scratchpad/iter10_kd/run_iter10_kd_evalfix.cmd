@echo off
REM ============================================================================
REM iter10_kd EVAL FIX (2026-07-13 18:10): the 2-parallel-shard eval wedged --
REM both shards silent 66+ min (last user completed ~17:03) at 11.7/12 GB VRAM,
REM 100%% GPU util, shard procs each burning a full CPU core = the iter-5 WDDM
REM oversubscription signature, this time on the CHAMPION arch (two mega-users
REM collided; clean champion evals finish entirely in ~90 min). Same fix as
REM run_lad_preset1_evalfix.cmd: run the SAME shard tomls SEQUENTIALLY
REM (get_result resumes -- completed users are skipped), then eval_sharded
REM relaunch-skip-merge, then the paired gate. Training artifacts (ckpt +
REM final cbs) were already done.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter10_kd\iter10_kd_evalfix.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=scratchpad/iter10_kd/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/iter10_kd/cb_shift_final.txt
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

echo ===== ITER10_KD EVALFIX START %DATE% %TIME% ===== > "%LOG%"
echo === SHARD 0 SEQUENTIAL (resume) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/eval_shards/shard_0.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_S0FAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
echo === SHARD 1 SEQUENTIAL (resume) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/eval_shards/shard_1.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_S1FAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
echo === MERGE (eval_sharded relaunch: shards skip all users, then merge) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter10_kd/iter10_kd_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_MERGEFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 4
)

echo === GATE: paired vs champ5k_b1 (iter 2) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-iter10_kd.jsonl --cand-imm result/RWKV-P-iter10_kd.jsonl --champ-ahead result/RWKV-champ5k_b1.jsonl --champ-imm result/RWKV-P-champ5k_b1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%: 0 = both p-gates pass) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
