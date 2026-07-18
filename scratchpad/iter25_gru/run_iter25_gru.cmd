@echo off
REM ============================================================================
REM RESEARCH ITER 25: GRU-faithful POWER-CURVE head at d=32 (RWKV_GRU_HEAD=2 --
REM three tiny linears predict per-row w/S/decay for N=2 power curves; replaces
REM the 64-basis exponential mixture; validated at d=128 by A3's deferred gate
REM PASS) + L0-v_lora free strip. 171,066 params (-11.7% vs iter 23). Full
REM iter-23 champion recipe incl. the PAVA rectifier (lambda=0.1, density=0.08).
REM State clamp ON as instability insurance (the GRU head destabilized d=128).
REM Gate: >=0.0003 BOTH modes vs iter 23 (0.304220/0.273423) + p<0.0001.
REM VPRUNE_MIN_STEP=6000 (zero-init GRU prior, the A3 lesson).
REM STEP 0 waits for the track-2 A5 run's DONE_EXIT (GPU handoff).
REM Launch DETACHED via detach.ps1 with ABSOLUTE path (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter25_gru\iter25_gru.log
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
set RWKV_GRU_HEAD=2
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STEP_TRACE=scratchpad/iter25_gru/iter25_gru_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000

echo ===== ITER25_GRU START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for track-2 A5 to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a5\track2_a5.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo A5 done -- starting iter 25 %TIME% >> "%LOG%"
del /Q scratchpad\iter25_gru\iter25_gru_ws_trace.jsonl scratchpad\iter25_gru\iter25_gru_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, d=32 GRU N=2 + strip + clamp + PAVA, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter25_gru/iter25_gru_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter25_gru iter25ws iter25d scratchpad/iter25_gru/iter25_gru_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter25_gru/iter25_gru_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter25_gru.jsonl result\RWKV-P-iter25_gru.jsonl result\RWKV-iter25_gru-solo.jsonl result\RWKV-P-iter25_gru-solo.jsonl result\RWKV-iter25_gru-s0.jsonl result\RWKV-P-iter25_gru-s0.jsonl result\RWKV-iter25_gru-s1.jsonl result\RWKV-P-iter25_gru-s1.jsonl result\RWKV-iter25_gru.nanskip.jsonl result\RWKV-iter25_gru-s0.nanskip.jsonl result\RWKV-iter25_gru-s1.nanskip.jsonl result\RWKV-iter25_gru-solo.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter25_gru iter25d scratchpad/iter25_gru/iter25_gru_eval.toml RWKV-iter25_gru RWKV-P-iter25_gru 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (phased sharded, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter25_gru/iter25_gru_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter23 champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter25_gru.jsonl --cand-imm result/RWKV-P-iter25_gru.jsonl --champ-ahead result/RWKV-iter23_pava.jsonl --champ-imm result/RWKV-P-iter23_pava.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
