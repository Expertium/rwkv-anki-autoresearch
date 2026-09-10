@echo off
REM w10clk -- THE LEAK-FREE DECAY BRANCH. Generated from run_w10plain.cmd (arm 1) by
REM scratchpad/w10clkdry/mk_w10clk.py; arm 1 is the exact control.
REM
REM THE LEVER, and it is the ONLY difference from arm 1: RWKV_CLOCK_AT_PREV_ANSWER=1800 in the decay,
REM its validation and the eval (rwkv/clock_fix.py). The query row (imm) and the PAVA probe rows
REM (rectified ahead) predict at SHOW time, and their clock columns carried the current review's
REM capped-duration excess; they are moved to the previous answer when the gap is in-session, and
REM the ahead label time loses the target review's in-session gap. Parameters are unchanged.
REM
REM A decay-only branch off a WS trained WITH the leak, so its number is a pessimistic estimate
REM of a fully leak-free model. PREREG: scratchpad/w10clkdry/PREREG.md.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10clkdry
set LOG=%DIR%\w10clkdry.log
set STAMP=%RANDOM%%RANDOM%
set TAG=w10clkdry
set STEPS=546
set WSSTEPS=109350

REM OMP_NUM_THREADS was sliced away when this env was copied from realcyc (it sits above the
REM RWKV_ block there). Without it each fetch worker runs torch on all 32 cores: ws10's two
REM workers showed 42-43 OS threads each and about 88 percent of the machine, for a fetch wait
REM that was already 3-7 ms and fully hidden. Pure oversubscription; found 2026-09-09.
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
REM The decay saves a checkpoint only on a validate step, and write_decay_setup hardcoded
REM 100000 -- so a 21,870-step decay checkpointed at step 50 and at the END and nowhere else.
REM A PC restart already killed one decay at 10,681/10,935. 2000 gives ~11 resume points for
REM ~4 percent of the phase, and it is trajectory-FREE: validation runs under model.eval()
REM inside no_grad and draws no main-process RNG (proven with a non-vacuity control by
REM scratchpad/ws10/rng_neutrality.py). Every endgame branch sets the SAME value, so they
REM stay mutually comparable.
set RWKV_DECAY_VALIDATE_EVERY=2000
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_MUON_INCLUDE_LORA=1
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1

REM ---- what distinguishes this arm: the -id layout AND the real-time cycles ----
set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
set RWKV_ZERO_FEATURES=
REM db overrides consumed by write_decay_setup.py / write_eval_toml.py, which used to
REM hardcode the 92-dim paths -- the trap the idfeat diagnostic caught.
set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_id5
set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_id5
set RWKV_LABEL_FILTER_DB=F:/rwkv_lmdb/label_filter_db_id_e2s
REM KD explicitly CLEARED, never merely unset, so an inherited value cannot turn the
REM control into a treatment -- the false-green shape that bit the rgate smoke.
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

REM ---- THE LEVER: the query-row clock-leak fix, in train, validation and eval ----
set RWKV_CLOCK_AT_PREV_ANSWER=1800


if not exist "%DIR%" mkdir "%DIR%"
echo ===== w10clkdry START %DATE% %TIME% ===== > "%LOG%"

REM ---- PHASE B: the 2-EPOCH decay from the shared 10-epoch WS checkpoint ----
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/ws10 w10_ws w10lkdry_d %DIR%\decay.toml F:/rwkv_lmdb/train_db_5k_h1_id5 1 5000 0.05 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% DSETUP_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_22 %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
findstr /C:"w10_ws_%WSSTEPS%" "%DIR%\dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% WRONGCKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_32 %DATE% %TIME% >> "%LOG%"
  exit /b 32
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config %DIR%\decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% DECAY_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_23 %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
REM The fix must have REACHED the fetch workers and RUN there. clock_fix prints its ON banner at
REM import in every process, and the chunk-count line only from apply_to_sample -- i.e. only if
REM prepare() really called it. A banner alone proves the flag parsed, never that it was used.
findstr /C:"ON: T=1800 s" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% CLOCK_FIX_NOT_ON %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_56 %DATE% %TIME% >> "%LOG%"
  exit /b 56
)
findstr /C:"s chunks 1 query rows" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% CLOCK_FIX_NEVER_RAN %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_57 %DATE% %TIME% >> "%LOG%"
  exit /b 57
)
REM KD must be ABSENT here -- the opposite of every recent runner's guard.
findstr /C:"[kd-mix] KD ON" "%DIR%\decay_%STAMP%.log" >nul
if %ERRORLEVEL%==0 (
  echo %TAG% KD_LEAKED_IN %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_43 %DATE% %TIME% >> "%LOG%"
  exit /b 43
)
if not exist "C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ws10\w10lkdry_d_%STEPS%.pth" (
  echo %TAG% DECAY_SHORT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_28 %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
echo %TAG% DECAY_OK %TIME% >> "%LOG%"

REM ---- PHASE C: rectified VAL-half eval, users 5001-7500 ----
set RWKV_EVAL_PAVA=1
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/w10clkdry w10lkdry_d %DIR%\eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 5020 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% ETOML_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_24 %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM The eval toml must point at THIS arm's db, or the arm silently scores on the
REM other arm's feature width. This is trap #2 made into a guard.
findstr /C:"F:/rwkv_lmdb/test_db_5k_id5" "%DIR%\eval.toml" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_DB_MISMATCH %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_34 %DATE% %TIME% >> "%LOG%"
  exit /b 34
)
if exist "result\RWKV-%TAG%.jsonl" del /q "result\RWKV-%TAG%.jsonl"
if exist "result\RWKV-P-%TAG%.jsonl" del /q "result\RWKV-P-%TAG%.jsonl"
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\eval.toml --shards 1 --solo-threshold 0 --exclude 6701 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_25 %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo %TAG% EVAL_OK %TIME% >> "%LOG%"
REM Non-fatal post-check: the eval's shard log must carry the ON banner too. Non-fatal on
REM purpose -- a guard that can destroy a finished ~3 h eval on a banner mismatch is worse than
REM what it guards; a marker file makes a miss impossible to overlook at verdict time.
findstr /C:"ON: T=1800 s" scratchpad\eval_shards\shard_s0.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_SHARD_LOG_HAS_NO_CLOCK_FIX_BANNER %DATE% %TIME% >> "%LOG%"
  echo clock-fix banner missing from the eval shard log > "%DIR%\EVAL_BANNER_MISSING.txt"
)
copy /y scratchpad\eval_shards\shard_s0.log "%DIR%\eval_shard_%STAMP%.log" >nul 2>&1

REM Terminal marker BEFORE endlocal: endlocal restores the pre-setlocal environment,
REM so %LOG% would expand to empty and the marker would go to "".
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0