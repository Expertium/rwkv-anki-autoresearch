@echo off
REM ============================================================================
REM RESEARCH ITER 29: hybrid Muon+AdamW optimizer (RWKV_MUON=1, rwkv/muon.py) on
REM the iter-26 champion recipe -- the modded-nanogpt sweep's one big transferable
REM (fresh optimizer family; matrices on Muon @ 0.02, rest on AdamW; matrix wd at
REM the AdamW-equivalent absolute rate). Gate (new bar): rounded-4dp >=0.0001
REM BOTH modes + p<0.0001 vs iter 26 (0.303942/0.273353).
REM STEP 0 waits for A8's DONE_EXIT; STEP 0.5 = 40-step E2E sanity (bench mode)
REM before committing to the full WS. VPRUNE_MIN_STEP=6000 (different optimizer
REM = different early dynamics). Launch DETACHED (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter29_muon\iter29_muon.log
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
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000

echo ===== ITER29_MUON START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for track-2 A8 to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a8\track2_a8.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo A8 done -- starting iter 29 %TIME% >> "%LOG%"

echo === STEP 0.5: 40-step E2E sanity (bench mode, muon wiring) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter29_muon/iter29_muon_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)
set RWKV_MAX_STEPS=

del /Q scratchpad\iter29_muon\iter29_muon_ws_trace.jsonl scratchpad\iter29_muon\iter29_muon_ws_trace.jsonl.val.jsonl 2>nul
set RWKV_STEP_TRACE=scratchpad/iter29_muon/iter29_muon_ws_trace.jsonl
echo === WS 1 epoch (1-5000, d=32 GRU N=3 + MUON, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter29_muon/iter29_muon_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter29_muon iter29ws iter29d scratchpad/iter29_muon/iter29_muon_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter29_muon/iter29_muon_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter29_muon.jsonl result\RWKV-P-iter29_muon.jsonl result\RWKV-iter29_muon-solo.jsonl result\RWKV-P-iter29_muon-solo.jsonl result\RWKV-iter29_muon-s0.jsonl result\RWKV-P-iter29_muon-s0.jsonl result\RWKV-iter29_muon-s1.jsonl result\RWKV-P-iter29_muon-s1.jsonl result\RWKV-iter29_muon.nanskip.jsonl result\RWKV-iter29_muon-s0.nanskip.jsonl result\RWKV-iter29_muon-s1.nanskip.jsonl result\RWKV-iter29_muon-solo.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter29_muon iter29d scratchpad/iter29_muon/iter29_muon_eval.toml RWKV-iter29_muon RWKV-P-iter29_muon 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (phased sharded, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter29_muon/iter29_muon_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter26 champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter29_muon.jsonl --cand-imm result/RWKV-P-iter29_muon.jsonl --champ-ahead result/RWKV-iter26_gru3.jsonl --champ-imm result/RWKV-P-iter26_gru3.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
