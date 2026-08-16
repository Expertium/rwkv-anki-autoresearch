@echo off
REM ===========================================================================================
REM ITER 51: POLAR-EXPRESS NEWTON-SCHULZ SCHEDULE FOR MUON (RWKV_MUON_POLAR=1). Family: optimizer.
REM PROPOSALS.md item #5; Andrew 2026-08-16 preferred the Muon variations over duration dropout.
REM
REM THE LEVER: Muon orthogonalizes its momentum with a quintic Newton-Schulz iteration acting on
REM each singular value as p(s) = a*s + b*s^3 + c*s^5. Production uses ONE fixed triple
REM (3.4445, -4.7750, 2.0315) for all five steps -- the modded-nanogpt constant, deliberately
REM sloppy: a+b+c = 0.7010, so it maps s=1 to 0.70 and lands the range in ~[0.70, 1.20]. This run
REM uses a PER-STEP schedule instead, fitted by greedy minimax on [0.0297, 1.0].
REM
REM MEASURED FIRST, ON REAL MOMENTUM BUFFERS (69 matrices from iter 50's optim state):
REM   PROPOSALS.md's 0.19-0.31 RMS orthogonality error REPRODUCES at 0.274 over all singular vals
REM   it is NOT precision:  bf16 0.289 vs fp32 0.301
REM   ~half is the near-null tail (median condition number 1.2e4) -- no odd polynomial lifts that
REM     in 5 steps, and lifting it would amplify noise, so it is not a target
REM   the ATTACKABLE part is 0.161 on the directions carrying the top 90 pct of momentum energy
REM   the fitted schedule cuts that 0.161 to 0.0251, a 84.4 pct reduction
REM   CONTROL: merely rescaling so sigma_max ~ 1 buys 5.8 pct, so the win is the SCHEDULE, not
REM     the input range. That control is why this is not just a normalisation bug.
REM
REM DOCUMENTED SIDE EFFECT: a more accurate polar factor changes update SIZE as well as shape --
REM ||O||_F rises 2.6 pct (median). Far below any LR sensitivity we have resolved (the tuner
REM needed 1.41-2.8x moves), so no compensating constant is folded in; one fitted to a single
REM checkpoint would be arbitrary. If this ACCEPTS by a thin margin, an LR control is the
REM follow-up.
REM
REM SINGLE VARIABLE vs the iter-45 champion: run_iter45.cmd with prefixes changed and ONE line
REM added (set RWKV_MUON_POLAR=1). Seed 4321, KD alpha 0.9 WS / 0.5 decay, PAVA 0.2, tuned HPs
REM and the speed stack all identical. Params UNCHANGED at 558,212 -- the lever is 5 constants.
REM
REM VERIFIED BEFORE LAUNCH: with the flag OFF, both the 2D and the BATCHED path are
REM BYTE-IDENTICAL to the pre-patch code on all 69 buffers (the loop was refactored, so this was
REM not free). The batched path is the live one -- RWKV_MUON_BATCHED=1 is in the standard env.
REM
REM DEPLOY: nothing to port. Training-only; no parameter, state, or forward-pass change.
REM
REM Do NOT git checkout / edit this file while it runs (iters 43 and 46 died that way).
REM NO del of result jsonls (fresh tag; retries resume from banked users).
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter51_muon
set LOG=%DIR%\iter51.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== ITER 51 (Muon polar-express NS schedule) START %DATE% %TIME% ===== > "%LOG%"

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
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
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

REM ================= THE LEVER, and the ONLY behavioural difference from run_iter45.cmd ========
set RWKV_MUON_POLAR=1

REM PHASE 0 GUARD -- the param count is CONSUMED state, unlike a banner. This lever changes FIVE
REM CONSTANTS inside the optimizer and must leave the MODEL byte-identical, so the expected count
REM is the champion's own 558,212. Anything else means an unrelated env var drifted in here.
.venv\Scripts\python.exe scratchpad\parity3\assert_param_count.py 558212 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_PARAMMISMATCH %DATE% %TIME% >> "%LOG%"
  exit /b 44
)

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter51_muon/i51_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
findstr /C:"[muon] POLAR-EXPRESS Newton-Schulz schedule ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOPOLAR_WS %DATE% %TIME% >> "%LOG%"
  exit /b 39
)
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
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter51_muon i51_ws i51_d scratchpad/iter51_muon/i51_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter51_muon/i51_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
findstr /C:"[muon] POLAR-EXPRESS Newton-Schulz schedule ON" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOPOLAR_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 40
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
echo DECAY OK (POLAR ON, KD ON, alpha 0.5) %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter51_muon i51_d scratchpad/iter51_muon/i51_eval.toml RWKV-iter51_muon RWKV-P-iter51_muon 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM RWKV_MUON_POLAR is irrelevant at eval -- Muon is an OPTIMIZER and does not exist in the
REM forward pass; it is left set only for uniformity. The speed stack IS cleared, as always, and
REM NO_JIT is cleared, so the eval SCRIPTS the model.
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter51_muon/i51_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter51_muon/i51_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
