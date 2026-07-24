@echo off
REM ============================================================================
REM BASELINE GRU v3 -- EVAL CONTINUATION (2026-07-25 00:45). The main .cmd's WS +
REM DECAY completed (bgrud_5586.pth); its EVAL phase WEDGED on its 11th user
REM (~11 GB VRAM reserved, 27 GB host working set, 0% GPU = WDDM oversubscription
REM spilling to host RAM) and was killed -> the control log carries
REM DONE_EXIT_EVALFAIL_1, which does NOT satisfy the LSTM's DONE_EXIT_0 waitloop,
REM so the chain is intact and this script re-runs eval + gate and appends
REM DONE_EXIT_0 to the SAME control log to release the parked LSTM.
REM
REM FIX: RWKV_EVAL_EMPTY_CACHE_EVERY=1 (new env in get_result.py; default 20 =
REM historical constant, RWKV runs byte-identical). fp32 RNN stream weights + the
REM per-layer probe tensors fragment the caching allocator far faster than a bf16
REM RWKV eval. Plus expandable_segments to cut fragmentation further.
REM eval_sharded RESUMES: users already in result/RWKV-baseline_gru-s0.jsonl
REM (5001-5010) are skipped.
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\baseline_gru
set LOG=%DIR%\baseline_gru.log
set STAMP=%RANDOM%%RANDOM%
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=0
set RWKV_AUGMENT_SEED=1234
set RWKV_NO_JIT=1
set RWKV_EXIT_HARD=1
set RWKV_BASELINE_CELL=gru
set RWKV_ARCH_MODULE=scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py
set RWKV_GRU_HEAD=2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_ZERO_FEATURES=22
set RWKV_EVAL_EMPTY_CACHE_EVERY=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo === EVAL RETRY (resume; empty_cache every user) %DATE% %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/baseline_gru/baseline_gru_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL2_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === COMPARISON vs A13 champion (INFORMATIONAL -- RWKV vs GRU at ~equal params) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-baseline_gru.jsonl --cand-imm result/RWKV-P-baseline_gru.jsonl --champ-ahead result/RWKV-track2_a13.jsonl --champ-imm result/RWKV-P-track2_a13.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%; baseline -- informational) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
