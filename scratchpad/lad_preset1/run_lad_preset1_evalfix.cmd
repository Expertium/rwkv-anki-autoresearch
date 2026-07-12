@echo off
REM ============================================================================
REM lad_preset1 EVAL FIX (2026-07-12 13:40): the original 2-parallel-shard eval
REM wedged -- both shards crawled 50-85+ min on their mega-users (whole run
REM would have been many hours vs iter-4's 92 min) with VRAM at 11.5/12 GB;
REM diagnosis = the preset-K=32 chunk-state buffers (~+0.8 GB/shard on 1M-token
REM batches) pushed TWO concurrent shards into WDDM oversubscription. Fix: run
REM the SAME two shard configs SEQUENTIALLY (one process at a time, ~6 GB peak;
REM get_result resumes -- users already in each shard's output jsonl are
REM skipped), then eval_sharded relaunches both (instant skip) and merges, then
REM the paired gate. Training artifacts (ckpt + final cbs) were already done.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\lad_preset1\lad_preset1_evalfix.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_STREAM_HEADS=preset:1
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=scratchpad/lad_preset1/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/lad_preset1/cb_shift_final.txt
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

echo ===== LAD_PRESET1 EVALFIX START %DATE% %TIME% ===== > "%LOG%"
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
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/lad_preset1/lad_preset1_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_MERGEFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 4
)

echo === GATE: paired vs champ5k_b1 (iter 2) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-lad_preset1.jsonl --cand-imm result/RWKV-P-lad_preset1.jsonl --champ-ahead result/RWKV-champ5k_b1.jsonl --champ-imm result/RWKV-P-champ5k_b1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%: 0 = both p-gates pass) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
