@echo off
REM ============================================================================
REM RESEARCH ITER 30: cautious weight decay on the Muon matrix groups
REM (RWKV_MUON_CAUTIOUS_WD=1, in-family sibling of the accepted iter-29 Muon;
REM modded-nanogpt #43/50: decay only coords whose applied step agrees with the
REM weight sign). Gate (val-split): rounded-4dp >=0.0001 BOTH modes + p<0.0001
REM vs iter 29 (val-half 0.302033/0.271440); eval VAL half 5001-7500 only.
REM GPU free at launch (iter 29 done) -- no waitloop. 40-step sanity then WS.
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter30_cwd\iter30_cwd.log
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
set RWKV_ZERO_FEATURES=22
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_GRU_HEAD=3
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_MUON=1
set RWKV_MUON_LR=0.02
set RWKV_MUON_MOMENTUM=0.95
set RWKV_MUON_CAUTIOUS_WD=1
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000

echo ===== ITER30_CWD START %DATE% %TIME% ===== > "%LOG%"

echo === STEP 0.5: 40-step E2E sanity (cautious-wd wiring) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter30_cwd/iter30_cwd_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)
set RWKV_MAX_STEPS=

del /Q scratchpad\iter30_cwd\iter30_cwd_ws_trace.jsonl scratchpad\iter30_cwd\iter30_cwd_ws_trace.jsonl.val.jsonl 2>nul
set RWKV_STEP_TRACE=scratchpad/iter30_cwd/iter30_cwd_ws_trace.jsonl
echo === WS 1 epoch (1-5000, d=32 GRU N=3 + MUON + CAUTIOUS WD, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter30_cwd/iter30_cwd_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.25 ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter30_cwd iter30ws iter30d scratchpad/iter30_cwd/iter30_cwd_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter30_cwd/iter30_cwd_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter30_cwd.jsonl result\RWKV-P-iter30_cwd.jsonl result\RWKV-iter30_cwd-solo.jsonl result\RWKV-P-iter30_cwd-solo.jsonl result\RWKV-iter30_cwd-s0.jsonl result\RWKV-P-iter30_cwd-s0.jsonl result\RWKV-iter30_cwd-s1.jsonl result\RWKV-P-iter30_cwd-s1.jsonl result\RWKV-iter30_cwd.nanskip.jsonl result\RWKV-iter30_cwd-s0.nanskip.jsonl result\RWKV-iter30_cwd-s1.nanskip.jsonl result\RWKV-iter30_cwd-solo.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter30_cwd iter30d scratchpad/iter30_cwd/iter30_cwd_eval.toml RWKV-iter30_cwd RWKV-P-iter30_cwd 5001 7500 >> "%LOG%" 2>&1
echo === EVAL (phased sharded, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter30_cwd/iter30_cwd_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter29 champion (val half) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter30_cwd.jsonl --cand-imm result/RWKV-P-iter30_cwd.jsonl --champ-ahead result/RWKV-iter29_muon.jsonl --champ-imm result/RWKV-P-iter29_muon.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
