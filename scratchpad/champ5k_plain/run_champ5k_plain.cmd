@echo off
REM ============================================================================
REM PLAIN RE-BASELINE (Andrew 2026-07-14): champ5k_plain = champ5k_b1's exact
REM recipe with NO QAT -- plain bf16, JIT on, no codebooks. Establishes the
REM plain-vs-plain screening champion + its vprune trace. Uses the NEW
REM power-user-aware eval_sharded (solo phase for users >= 1M work, then 2
REM parallel shards, then merge -- one call, resume-safe). No vprune (this IS
REM the new reference). Final paired_pvalue vs champ5k_b1 is INFORMATIONAL
REM (measures the QAT tax at n=5000), not a gate. Launch DETACHED (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\champ5k_plain\champ5k_plain.log
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
set RWKV_STEP_TRACE=scratchpad/champ5k_plain/champ5k_plain_ws_trace.jsonl

echo ===== CHAMP5K_PLAIN START %DATE% %TIME% ===== > "%LOG%"
del /Q scratchpad\champ5k_plain\champ5k_plain_ws_trace.jsonl scratchpad\champ5k_plain\champ5k_plain_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, PLAIN, no vprune) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/champ5k_plain/champ5k_plain_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.25 ep = ratio 0.25 of 1 WS ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/champ5k_plain champ5kplainws champ5kplaind scratchpad/champ5k_plain/champ5k_plain_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/champ5k_plain/champ5k_plain_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-champ5k_plain.jsonl result\RWKV-P-champ5k_plain.jsonl result\RWKV-champ5k_plain-solo.jsonl result\RWKV-P-champ5k_plain-solo.jsonl result\RWKV-champ5k_plain-s0.jsonl result\RWKV-P-champ5k_plain-s0.jsonl result\RWKV-champ5k_plain-s1.jsonl result\RWKV-P-champ5k_plain-s1.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/champ5k_plain champ5kplaind scratchpad/champ5k_plain/champ5k_plain_eval.toml RWKV-champ5k_plain RWKV-P-champ5k_plain 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (power-user-aware: solo phase then 2 parallel shards then merge) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/champ5k_plain/champ5k_plain_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === INFO: paired vs champ5k_b1 (QAT tax measurement, NOT a gate) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-champ5k_plain.jsonl --cand-imm result/RWKV-P-champ5k_plain.jsonl --champ-ahead result/RWKV-champ5k_b1.jsonl --champ-imm result/RWKV-P-champ5k_b1.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (informational paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
