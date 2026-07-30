@echo off
REM Eval-only re-run: WS and decay COMPLETED (mvd_2733.pth); only the eval died.
REM CAUSE: marginal GPU OOM on user 5995 (266,435 reviews) -- VRAM hit 11,814 of 12,282 MiB
REM at 14:21:37 and the process was gone by 14:22:20. That user evaluated fine in the three
REM previous evals, and nothing unusual held VRAM, so it is a coin flip at the card's edge.
REM MITIGATION: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- reduces allocator
REM fragmentation, which is what decides a marginal OOM. Allocator strategy only, so it
REM cannot change numerics.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\maxval
set LOG=%DIR%\maxval_eval3.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
REM expandable_segments REMOVED 2026-07-30: the evals that PASSED the giant users never used it
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_EVAL_PAVA=1

echo ===== MAXVAL EVAL RETRY START %DATE% %TIME% ===== > "%LOG%"
REM RESUME MODE (2026-07-30): no del -- eval_sharded SKIPS completed users, and the
REM giant-user OOMs are intermittent, so every banked user shrinks the next attempt's risk.

.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/maxval/maxval_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval_retry_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === GATE: maxval vs iter 31 RECTIFIED %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-maxval.jsonl --cand-imm result/RWKV-P-maxval.jsonl --champ-ahead result/RWKV-iter31_algo_rect.jsonl --champ-imm result/RWKV-P-iter31_algo_rect.jsonl > "%DIR%\gate_retry_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
