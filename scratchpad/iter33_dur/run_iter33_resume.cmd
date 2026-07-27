@echo off
REM ============================================================================
REM iter 33 RESUME from WS step 14,000 of 43,354 (stopped 2026-07-27 17:38 at
REM Andrew's request for cable management; not a failure).
REM
REM Differences from run_iter33_dur.cmd, all deliberate:
REM   * no waitloop (nothing to wait for) and no 40-step sanity phase (it passed
REM     at 09:53 and the VRAM envelope is already proven at MAX=16384/density 1.0)
REM   * RWKV_RESUME_SKIP_GROUPS=1 + WS pointed at iter33_dur_ws_resume.toml
REM   * the step-trace files are NOT deleted -- the trace continues from 14,934
REM   * GATE now pairs against iter 32 RECTIFIED, not iter 31: iter 32 was
REM     promoted to champion 2026-07-27 23:13 after its rectified eval landed
REM     (+0.000534 ahead / +0.000429 imm vs iter 31, both p<1e-50).
REM   * banner says vprune OFF, which is what the env actually does -- the
REM     original printed "vprune ON min6000" from a stale template string.
REM
REM ⚠ RWKV_MUON_BATCHED IS DELIBERATELY NOT SET. The batched Newton-Schulz
REM (committed 13fd1b1) is a real speedup but it perturbs optimizer numerics on
REM GPU, and the first 14,000 steps were trained WITHOUT it. Leaving it off keeps
REM the two halves of this run consistent; the speedup applies to the NEXT run.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter33_dur
set LOG=%DIR%\iter33_dur.log
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
set RWKV_PROBE_DENSITY=1.0
set RWKV_AHEAD_PROBE_ONLY=1
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.02
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_RESUME_SKIP_GROUPS=1

echo ===== iter33_dur RESUME from step 14000 START %DATE% %TIME% ===== >> "%LOG%"

set RWKV_STEP_TRACE=scratchpad/iter33_dur/iter33_dur_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/iter33_dur/iter33_grad_stats_ws_resume.json

echo === WS resume 14001-43354 (vprune OFF, MUON_BATCHED off for consistency) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter33_dur/iter33_dur_ws_resume.toml > "%DIR%\ws_resume_%STAMP%.log" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
echo WS OK %TIME% >> "%LOG%"
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/iter33_dur/iter33_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=16384) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter33_dur iter33ws iter33d scratchpad/iter33_dur/iter33_dur_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 16384 > "%DIR%\decay_setup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter33_dur/iter33_dur_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo DECAY OK %TIME% >> "%LOG%"

del /Q result\RWKV-iter33_dur.jsonl result\RWKV-P-iter33_dur.jsonl result\RWKV-iter33_dur-s0.jsonl result\RWKV-P-iter33_dur-s0.jsonl result\RWKV-iter33_dur.nanskip.jsonl result\RWKV-iter33_dur-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter33_dur iter33d scratchpad/iter33_dur/iter33_dur_eval.toml RWKV-iter33_dur RWKV-P-iter33_dur 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
set RWKV_EVAL_PAVA=1
echo === EVAL (RECTIFIED, single process, d=80, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter33_dur/iter33_dur_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === GATE: paired vs CHAMPION iter32 RECTIFIED (val half) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter33_dur.jsonl --cand-imm result/RWKV-P-iter33_dur.jsonl --champ-ahead result/RWKV-iter32_kd_rect.jsonl --champ-imm result/RWKV-P-iter32_kd_rect.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
