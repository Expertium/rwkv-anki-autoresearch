@echo off
REM ===========================================================================================
REM ITER 49: RESTORE THE user/preset LAYER-0 CHANNEL MIXERS. Family: capacity.
REM PROPOSALS.md item #6, top of the untried list now that iter 48 closed the ahead-vs-imm family.
REM ZERO CODE: the lever is a SHORTER RWKV_STRIP_CMIX list, restoring the two layer-0 channel
REM mixers that A6 stripped from the user and preset streams on grad-stats bottom-saliency.
REM
REM WHY NOW: the capacity family is UNTESTED on this trunk. Its only evidence is the 100-user d=32
REM era, where every capacity add was rejected and the conclusion was 'DATA-limited, not
REM capacity-limited'. That was measured at 100 users; this trunk trains on 5,000, so the premise
REM that verdict rested on no longer holds and the question is genuinely open.
REM
REM COST: +26,070 params (558,212 to 584,282, +4.7 pct). The old params 225k cap is RETIRED (it
REM belonged to the d=32 track). Gate #3 is safe BY CONSTRUCTION: the strip list changes only
REM user_id/preset_id entries -- card_id:1, deck_id:1, deck_id:2 are identical in both lists -- so
REM card and note state are untouched, and deck/preset/global MAY grow freely.
REM
REM SINGLE VARIABLE vs the iter-45 champion: run_iter45.cmd with save prefixes changed and the
REM strip list shortened. Seed 4321, KD alpha 0.9 WS / 0.5 decay, PAVA 0.2, tuned HPs identical.
REM
REM DEPLOY: forward-pass shape change, so it needs a fresh parity trace BEFORE shipping -- but only
REM if ACCEPTED. The Rust engine already reads per-layer cmix skips from the checkpoint.
REM
REM ⚠ Do NOT git checkout / edit this file while it runs (iters 43 and 46 died that way; git
REM normalises line endings, which shifts cmd.exe's saved byte offset).
REM ⚠ NO del of result jsonls (fresh tag; retries resume from banked users).
REM ⚠ No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter49_rcouple
set LOG=%DIR%\iter49.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== ITER 49 (retrievability-coupled rating head) START %DATE% %TIME% ===== > "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
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
set RWKV_STRIP_CMIX=user_id:1,user_id:2,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_GRAD_STATS=%DIR%\grad_stats.json
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.9

REM ===== THE LEVER: the RWKV_STRIP_CMIX line above drops user_id:0 and preset_id:0, i.e. it =====
REM ===== RESTORES those two layer-0 channel mixers. Zero code.                             =====
REM PHASE 0 GUARD -- assert the env really built the intended model. RWKV_STRIP_CMIX prints NO
REM banner, and the QAT-inert bug proved a banner is the wrong check even when one exists (it
REM reported a PARSED value for an object discarded one line later). The PARAM COUNT is CONSUMED
REM state: champion 558,212 vs restored 584,282 (+26,070, +4.7 pct). A typo'd stream name lands
REM here, before any GPU is spent.
.venv\Scripts\python.exe scratchpad\parity3\assert_param_count.py 584282 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_PARAMMISMATCH %DATE% %TIME% >> "%LOG%"
  exit /b 44
)

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter49_cmix/i49_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
REM THE experiment's own guard: the coupling must actually be ON. A banner proves the value was
REM COMPUTED, so this is necessary but not sufficient -- the sufficiency check is that
REM smoke_rcouple.py asserts the coupling MOVES P(Again) (the qat-inert lesson: a truthful banner
REM on an object that is then discarded). Without this the run silently reduces to iter 45 and
REM would be recorded as a null.
findstr /C:"[interleave] round-robin layer schedule ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOILV %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
findstr /C:"placement = front-loaded" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGPLACEMENT %DATE% %TIME% >> "%LOG%"
  exit /b 33
)
echo WS OK %TIME% >> "%LOG%"

set RWKV_KD_ALPHA=0.5
set RWKV_GRAD_STATS=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter49_cmix i49_ws i49_d scratchpad/iter49_cmix/i49_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter49_cmix/i49_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
findstr /C:"[kd-mix] KD ON" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOKD_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 35
)
findstr /C:"alpha FIXED at 0.5" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGALPHA_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 36
)
echo DECAY OK (cmix restored, KD ON, alpha 0.5) %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter49_cmix i49_d scratchpad/iter49_cmix/i49_eval.toml RWKV-iter49_rcouple RWKV-P-iter49_rcouple 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM ⚠ RWKV_STRIP_CMIX stays SET for the eval -- it is a MODEL SHAPE property. Clearing it would
REM build a different architecture and fail the strict checkpoint load outright. It is deliberately
REM absent from the clear-list below; only the speed stack and KD are cleared.
REM ⚠ NO_JIT is cleared here, so the eval SCRIPTS the model -- the ONLY path that does, and the
REM one iter 48's eval died on (an unannotated @torch.jit.ignore returning None). Fixed and
REM exercised end-to-end by iter 48's re-run over all 2,500 users, so this path is now proven --
REM not merely 'compiles', which is what that claim wrongly rested on the first time.
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter49_cmix/i49_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter49_cmix/i49_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
