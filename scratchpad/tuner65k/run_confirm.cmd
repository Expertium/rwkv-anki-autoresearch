@echo off
REM ============================================================================
REM FULL VAL-HALF CONFIRMATION of the HP tuner winner (iter 34's actual gate).
REM
REM The tuner ranks on the 1000-user subset 5001-6000. That is a RANKING PROXY:
REM the champ5k_t1 lesson showed a 200-user subset winner INVERT at full scale,
REM and while this 1000-user subset is much better behaved (it ranked
REM maxval-vs-iter-31 the same way the full half did), a sub-0.001 verdict still
REM has to be confirmed on 5001-7500 before it becomes the recipe.
REM
REM NO RETRAINING: the winning trial's decay checkpoint is already on disk, so
REM this is one eval (~2.5 h) plus a free paired comparison.
REM
REM Env note: this sets the A18 trunk env WITHOUT the three training speed flags
REM (RWKV_MUON_BATCHED / RWKV_NO_JIT / RWKV_QAT_COMPILE), exactly as every trial
REM .cmd clears them before its own eval.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner65k
set LOG=%DIR%\confirm.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
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
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_EVAL_PAVA=1

echo ===== CONFIRM (full VAL half 5001-7500, RECTIFIED) START %DATE% %TIME% ===== > "%LOG%"

echo === RESOLVE WINNER + WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/confirm_winner.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_RESOLVEFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 3
)

REM Giant users 5002/5905/5995 can OOM if the desktop holds VRAM. eval_sharded SKIPS
REM users already banked, so three attempts with NO del between them only re-risk the
REM remainder (the 2026-07-30 big-eval ops rule).
echo === EVAL attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/confirm_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\confirm_eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/confirm_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\confirm_eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo EVAL attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/confirm_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\confirm_eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === GATE: winner vs iter 32 RECTIFIED (rect-vs-rect, the iter-33-on basis) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-t65confirm.jsonl --cand-imm result/RWKV-P-t65confirm.jsonl --champ-ahead result/RWKV-iter32_kd_rect.jsonl --champ-imm result/RWKV-P-iter32_kd_rect.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
