@echo off
REM ============================================================================
REM RESEARCH ITER 27: GRU head N=4 (sweep continues) (RWKV_GRU_HEAD=4) on the iter-25 champion
REM recipe (Andrews standing directive after accepting iter 25). Gate: >=0.0003
REM BOTH modes vs iter 25 (0.304427/0.273441) + p<0.0001. vprune vs
REM champion_5k_plain.json (= iter 25s traces), MIN_STEP=6000 (zero-init prior).
REM STEP 0 waits for meme_blind's DONE_EXIT. Launch DETACHED (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter27_gru4\iter27_gru4.log
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
set RWKV_GRU_HEAD=4
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STEP_TRACE=scratchpad/iter27_gru4/iter27_gru4_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000

echo ===== iter27_gru4 START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for iter 26 to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter26_gru3\iter26_gru3.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo iter 26 done -- starting iter 27 %TIME% >> "%LOG%"
del /Q scratchpad\iter27_gru4\iter27_gru4_ws_trace.jsonl scratchpad\iter27_gru4\iter27_gru4_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, d=32 GRU N=3, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter27_gru4/iter27_gru4_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter27_gru4 iter27ws iter27d scratchpad/iter27_gru4/iter27_gru4_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter27_gru4/iter27_gru4_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter27_gru4.jsonl result\RWKV-P-iter27_gru4.jsonl result\RWKV-iter27_gru4-solo.jsonl result\RWKV-P-iter27_gru4-solo.jsonl result\RWKV-iter27_gru4-s0.jsonl result\RWKV-P-iter27_gru4-s0.jsonl result\RWKV-iter27_gru4-s1.jsonl result\RWKV-P-iter27_gru4-s1.jsonl result\RWKV-iter27_gru4.nanskip.jsonl result\RWKV-iter27_gru4-s0.nanskip.jsonl result\RWKV-iter27_gru4-s1.nanskip.jsonl result\RWKV-iter27_gru4-solo.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter27_gru4 iter27d scratchpad/iter27_gru4/iter27_gru4_eval.toml RWKV-iter27_gru4 RWKV-P-iter27_gru4 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (phased sharded, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter27_gru4/iter27_gru4_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter25 champion %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter27_gru4.jsonl --cand-imm result/RWKV-P-iter27_gru4.jsonl --champ-ahead result/RWKV-iter25_gru.jsonl --champ-imm result/RWKV-P-iter25_gru.jsonl >> "%LOG%" 2>&1
echo === GATE-B: paired vs iter26 (in case Andrew promotes N=3) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter27_gru4.jsonl --cand-imm result/RWKV-P-iter27_gru4.jsonl --champ-ahead result/RWKV-iter26_gru3.jsonl --champ-imm result/RWKV-P-iter26_gru3.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
