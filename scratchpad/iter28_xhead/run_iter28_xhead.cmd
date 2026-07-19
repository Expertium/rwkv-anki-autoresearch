@echo off
REM ============================================================================
REM RESEARCH ITER 28: xhead-mix v1 RE-BENCHMARK on the current champion recipe
REM (Andrew 2026-07-19: iter 20s +0.000178/+0.000107 at p 2e-10/2e-25 would pass
REM the NEW rounded-4dp >=0.0001 gate, but was measured vs the stale iter-15
REM recipe -- re-earn it on GRU N=3 + PAVA + no-residual + strip + clamp).
REM RWKV_XHEAD_MIX=1: zero-init per-channel (H,H,K) cross-head delta on the WKV
REM output pre-GroupNorm, +896 params. Gate (NEW bar): rounded-4dp >=0.0001 BOTH
REM modes + p<0.0001 vs the CURRENT champion at record time (tail prints paired
REM vs BOTH iter 26 and iter 27). VPRUNE_MIN_STEP=6000 (GRU prior).
REM STEP 0 waits for track-2 A6's DONE_EXIT. Launch DETACHED (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter28_xhead\iter28_xhead.log
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
set RWKV_XHEAD_MIX=1
set RWKV_STEP_TRACE=scratchpad/iter28_xhead/iter28_xhead_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k_plain.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000

echo ===== ITER28_XHEAD START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for track-2 A6 to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a6\track2_a6.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo A6 done -- starting iter 28 %TIME% >> "%LOG%"
del /Q scratchpad\iter28_xhead\iter28_xhead_ws_trace.jsonl scratchpad\iter28_xhead\iter28_xhead_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, d=32 GRU N=3 + xhead v1, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter28_xhead/iter28_xhead_ws.toml >> "%LOG%" 2>&1
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter28_xhead iter28ws iter28d scratchpad/iter28_xhead/iter28_xhead_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter28_xhead/iter28_xhead_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-iter28_xhead.jsonl result\RWKV-P-iter28_xhead.jsonl result\RWKV-iter28_xhead-solo.jsonl result\RWKV-P-iter28_xhead-solo.jsonl result\RWKV-iter28_xhead-s0.jsonl result\RWKV-P-iter28_xhead-s0.jsonl result\RWKV-iter28_xhead-s1.jsonl result\RWKV-P-iter28_xhead-s1.jsonl result\RWKV-iter28_xhead.nanskip.jsonl result\RWKV-iter28_xhead-s0.nanskip.jsonl result\RWKV-iter28_xhead-s1.nanskip.jsonl result\RWKV-iter28_xhead-solo.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter28_xhead iter28d scratchpad/iter28_xhead/iter28_xhead_eval.toml RWKV-iter28_xhead RWKV-P-iter28_xhead 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (phased sharded, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter28_xhead/iter28_xhead_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === GATE: paired vs iter26 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter28_xhead.jsonl --cand-imm result/RWKV-P-iter28_xhead.jsonl --champ-ahead result/RWKV-iter26_gru3.jsonl --champ-imm result/RWKV-P-iter26_gru3.jsonl >> "%LOG%" 2>&1
echo === GATE-B: paired vs iter27 (in case N=4 is champion) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter28_xhead.jsonl --cand-imm result/RWKV-P-iter28_xhead.jsonl --champ-ahead result/RWKV-iter27_gru4.jsonl --champ-imm result/RWKV-P-iter27_gru4.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
