@echo off
REM ===========================================================================================
REM ITER 43 PHASE-2 RESUME (2026-08-10 03:0x): DECAY + RECTIFIED EVAL only.
REM
REM WHY THIS EXISTS: the original run_iter43.cmd's WS phase COMPLETED normally
REM (i43_ws_10935.pth + grad_stats.json written 02:54:37), but the chain then died at 02:55:04
REM with 'ratchpad' is not recognized / DONE_EXIT_WSFAIL_9009 -- cmd.exe's saved read-offset
REM corruption. CAUSE: run_iter43.cmd was committed AFTER launch and then rewritten on disk by
REM `git rebase --autostash`, while cmd.exe was still executing it. cmd.exe re-reads the batch
REM file from a byte offset after each command finishes, so the rewrite made it resume
REM mid-token ("scratchpad" -> "ratchpad"). The 9009 is NOT a training failure.
REM LESSON (same class as CLAUDE.md's "editing a live .cmd corrupts cmd.exe's saved read
REM offset"): ANY git operation that rewrites a running .cmd -- commit is fine, but
REM rebase/checkout/stash rewrite the working tree -- is the same hazard. Do not touch git on a
REM running runner's path until its chain reports DONE_EXIT.
REM COLLATERAL: the WS log was truncated to 99 B by the garbage line's redirect (same %STAMP%,
REM same cmd.exe process), so iter 43 has NO ws step trace to extract a val trace from. The
REM checkpoints and grad_stats survive; only the log text is gone.
REM
REM Guards: assert the FINAL WS ckpt exists (write_decay_setup takes the LATEST ckpt and would
REM silently decay a half-trained one), and the decay log must carry the interleave banner and
REM must NOT name the _cnd arch.
REM ⚠ NO del of result jsonls (fresh tags; retries resume from banked users).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter43_ilv_old
set LOG=%DIR%\iter43.log
set STAMP=%RANDOM%%RANDOM%

echo ===== ITER 43 PHASE-2 (decay + eval) START %DATE% %TIME% ===== >> "%LOG%"

if not exist "%DIR%\i43_ws_10935.pth" (
  echo DONE_EXIT_NOFINALCKPT %DATE% %TIME% >> "%LOG%"
  exit /b 31
)
echo WS final ckpt present (i43_ws_10935.pth) %TIME% >> "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
REM ---- THE LEVER: the ORIGINAL order, not the _cnd reorder ----
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
REM the adopted speed stack (training only; cleared before eval). No KD, no GRAD_STATS: decay
REM never had them (the original .cmd cleared both before this phase).
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter43_ilv_old i43_ws i43_d scratchpad/iter43_ilv_old/i43_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
findstr /C:"i43_ws_10935" "%DIR%\dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGCKPT %DATE% %TIME% >> "%LOG%"
  exit /b 32
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter43_ilv_old/i43_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
findstr /C:"[interleave] round-robin layer schedule ON" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOILV_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 29
)
findstr /C:"architecture_d80_lora4_cnd.py" "%DIR%\decay_%STAMP%.log" >nul
if %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGARCH %DATE% %TIME% >> "%LOG%"
  exit /b 30
)
echo DECAY OK %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter43_ilv_old i43_d scratchpad/iter43_ilv_old/i43_eval.toml RWKV-iter43_ilvold RWKV-P-iter43_ilvold 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM eval without the training speed flags, rectified, unsharded (d=80 rule).
REM ⚠ RWKV_INTERLEAVE stays SET -- eval must score the interleaved model.
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter43_ilv_old/i43_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter43_ilv_old/i43_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter43_ilv_old/i43_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
