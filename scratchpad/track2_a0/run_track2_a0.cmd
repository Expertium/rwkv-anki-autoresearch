@echo off
REM ============================================================================
REM TRACK 2 ANCHOR A0: original d=128 arch (2,762,884 params) retrained through
REM the current PLAIN pipeline (1 ep WS + 0.25 ep decay, seed 1234, MAX=66000).
REM The "before" anchor for the ablation ratio gate. Eval = ONE process (d=128
REM cannot share the 12 GB GPU). Final paired vs the upstream 12-ep base5k
REM result = the 1-ep-budget check at 14x params (INFORMATIONAL, not a gate).
REM ~13-14 h total. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a0\track2_a0.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
REM Clear EVERY step over the WHOLE run (WINDOW=0). Launch 4 (guard = first 1000 steps
REM only) crept 3.6->11.3 GB by step ~4100 (allocator envelope over variable d=128 group
REM shapes) -> WDDM paging, 4.3 s/step. Launch 5 (every=50) saturated 11.9/12 GB by step
REM ~250. Per-step clears hold 3.6 GB at an unchanged ~1.06 s/step (the ~0.1 s clear
REM hides under the ~1 s d=128 step; launch-4 steps 1-1000 prove it). Numerics-neutral.
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_STEP_TRACE=scratchpad/track2_a0/track2_a0_ws_trace.jsonl

echo ===== TRACK2_A0 START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\track2_a0\track2_a0_ws_trace.jsonl scratchpad\track2_a0\track2_a0_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, d=128 PLAIN, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a0/track2_a0_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.25 ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a0 t2a0ws t2a0d scratchpad/track2_a0/track2_a0_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a0/track2_a0_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-track2_a0.jsonl result\RWKV-P-track2_a0.jsonl result\RWKV-track2_a0-s0.jsonl result\RWKV-P-track2_a0-s0.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a0 t2a0d scratchpad/track2_a0/track2_a0_eval.toml RWKV-track2_a0 RWKV-P-track2_a0 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (single process, d=128 unshardable) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a0/track2_a0_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === INFO: paired vs upstream 12-ep d=128 (budget check, NOT a gate) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-track2_a0.jsonl --cand-imm result/RWKV-P-track2_a0.jsonl --champ-ahead result/RWKV-base5k.jsonl --champ-imm result/RWKV-P-base5k.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (informational paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
