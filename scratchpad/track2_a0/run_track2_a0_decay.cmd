@echo off
REM ============================================================================
REM TRACK 2 A0 -- RESUME FROM THE DECAY PHASE (2026-07-15). The 6.7h WS finished
REM clean (t2a0ws_22346.pth); the first decay attempt thrashed because
REM write_decay_setup defaulted MAX_TRAIN_GLOBAL_LEN=110000 (d=32 standard) --
REM d=128 needs the track-2 standard 32768 (now passed as arg 10). This .cmd
REM re-runs decay-setup + decay + eval + the informational paired test, appending
REM to the same log. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a0\track2_a0.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25

echo ===== TRACK2_A0 DECAY-RESUME START %DATE% %TIME% ===== >> "%LOG%"
echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a0 t2a0ws t2a0d scratchpad/track2_a0/track2_a0_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 >> "%LOG%" 2>&1
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
