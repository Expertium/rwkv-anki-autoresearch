@echo off
REM ===========================================================================================
REM ITER 45: KD THROUGH THE DECAY PHASE (2026-08-10, family: distillation, 3/3 accepted).
REM Winner of the 15-proposal ranking. ZERO CODE CHANGE -- the entire lever is that this runner
REM does NOT clear RWKV_KD_MIX/RWKV_KD_ALPHA before the decay phase, and sets a decay-specific
REM alpha instead.
REM
REM WHY: since iter 34 adopted decay_ratio=1.0, the decay phase is HALF of all training, and it
REM runs on pure hard labels -- the champion runner clears KD before it. Meanwhile the WS alpha
REM dose curve is monotone up to 0.9 (iters 32/35/38/39/40), i.e. this model wants teacher signal
REM badly. So half of training currently gets none, in the only family with a perfect record.
REM Iter 40's own note lists "KD-through-decay" as the open in-family item.
REM
REM WHY IT IS WELL-DEFINED (the belief that the dump cannot be used in decay is wrong, and our own
REM docs contain the reason): decay writes STEP_OFFSET=1 and EPOCHS=1.0 over the SAME db/MAX/seed,
REM and `random.seed(12345)` + a single get_groups shuffle mean decay step s replays byte-identically
REM the batch of WS step s (CLAUDE.md: "epochs are BYTE-IDENTICAL replays"). Verified for this run:
REM i44_decay.toml has STEP_OFFSET=1, EPOCHS=1.0, MAX=65536; the dump has exactly 10,935 files
REM step_1.pt..step_10935.pt; decay is 10,935 steps at ratio 1.0. The alignment is EXACT.
REM ⚠ CLAUDE.md warns "the checksum cannot catch KD misfiring [in decay]" -- that warning is about
REM leaving KD on by ACCIDENT, and its premise (decay reproduces the epoch-0 stream) is precisely
REM what makes doing it DELIBERATELY correct.
REM
REM SELF-VALIDATING: train_rwkv.py:1252-1266 re-checks a per-step labels_sum + shape against the
REM dump and hard-exits 43 on any mismatch. If the streams do NOT align, this aborts at step 1
REM rather than training on mismatched targets. There is no silent-corruption mode.
REM
REM ALPHA: WS keeps its tuned 0.9; decay gets 0.5. Lower because decay's job is annealing onto the
REM true objective and iter 40 showed removing hard labels entirely (alpha 1.0) costs imm. 0.5 also
REM has a track record -- it is iter 32's accepted WS value. If this wins, alpha_decay=0.9 is a
REM free second point on the same dump.
REM
REM BUDGET: set MAXSTEPS below to 0 for the full budget (gate vs RWKV-iter41_ilv-s0.jsonl) or to
REM 3645 for the 1/3 budget (gate vs the calibration's RWKV-c41.jsonl, which IS this same champion
REM recipe at that budget). Decide AFTER the budget calibration reports.
REM ⚠ Do NOT git rebase/pull/checkout this path while it runs (iter 43's chain died that way).
REM ⚠ NO del of result jsonls (fresh tags; retries resume from banked users).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter45_kddecay
set LOG=%DIR%\iter45.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935
REM ---- BUDGET: 0 = full, 3645 = 1/3 ----
set MAXSTEPS=0

echo ===== ITER 45 (KD through decay) START %DATE% %TIME% (maxsteps=%MAXSTEPS%) ===== > "%LOG%"

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
if not "%MAXSTEPS%"=="0" set RWKV_MAX_STEPS=%MAXSTEPS%
if not "%MAXSTEPS%"=="0" set RWKV_BENCH_WARMUP=100000

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter45_kddecay/i45_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
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

REM ---- THE LEVER: KD is NOT cleared. Only the alpha changes for the decay phase. ----
set RWKV_KD_ALPHA=0.5
set RWKV_GRAD_STATS=
set RWKV_MAX_STEPS=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter45_kddecay i45_ws i45_d scratchpad/iter45_kddecay/i45_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter45_kddecay/i45_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
REM THE experiment's own guard: KD must actually have been ON during decay. Without this the run
REM silently reduces to the champion and would be recorded as a null.
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
echo DECAY OK (KD ON, alpha 0.5) %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter45_kddecay i45_d scratchpad/iter45_kddecay/i45_eval.toml RWKV-iter45_kddecay RWKV-P-iter45_kddecay 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter45_kddecay/i45_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter45_kddecay/i45_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
