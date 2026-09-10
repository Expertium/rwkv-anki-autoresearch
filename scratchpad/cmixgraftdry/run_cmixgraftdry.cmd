@echo off
REM cmixgraft -- COARSE-STREAM CAPACITY AT THE REAL BUDGET. Generated from run_w10plain.cmd
REM (arm 1) by scratchpad/cmixgraftdry/mk_cmixgraft.py; arm 1 is the exact control.
REM
REM THE LEVER, and it is the ONLY difference from arm 1: RWKV_STRIP_CMIX drops the six
REM user_id/preset_id entries, restoring those channel mixers. 563,652 -> 641,862 params.
REM Those streams are per-deck and per-user, so the 9 B/card and 27 B/note deploy contract is
REM untouched -- gate 3 has always allowed deck/preset/global to grow.
REM
REM WHY IT COSTS A DECAY AND NOT A 38 h WS PHASE. The graft is function-preserving at init: a
REM channel mixer returns in + dropout(W_v(...)) and RWKV7ChannelMixer zeroes W_v at construction.
REM scratchpad/cmixgraftdry/expand_ckpt.py writes w10g_ws_109350.pth (ws10's WS-final with the six
REM mixers grown from their d_model=1 dummy shapes), and smoke_graft.py measures 0.000e+00 against
REM arm 1 over 300 deploy predictions with a PERTURBED control at 1.132e-06 to prove non-vacuity.
REM
REM ANDREW 2026-09-10: "It would be nice to try to grow deck/preset/global, but 38 h is a pretty
REM steep price" and "I'd rather not increase card/note state size". Both are satisfied.
REM
REM PREREG: scratchpad/cmixgraftdry/PREREG.md -- P1 ahead >= +0.0003, P3 the restored W_v must be
REM non-zero at the end or the run is uninterpretable, and the asymmetry: a WIN is decisive, a
REM NULL is weak. It also prices the one second variable, p=0.005 layer dropout on six blocks.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\cmixgraftdry
set LOG=%DIR%\cmixgraftdry.log
set STAMP=%RANDOM%%RANDOM%
set TAG=cmixgraftdry
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
set RWKV_STRIP_CMIX=deck_id:1,deck_id:2,card_id:1
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


if not exist "%DIR%" mkdir "%DIR%"
echo ===== cmixgraftdry START %DATE% %TIME% ===== > "%LOG%"

REM ---- PHASE B: the 2-EPOCH decay from the shared 10-epoch WS checkpoint ----
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/ws10 w10g_ws w10gdry_d %DIR%\decay.toml F:/rwkv_lmdb/train_db_5k_h1_id5 1 5000 0.05 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% DSETUP_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_22 %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
findstr /C:"w10g_ws_%WSSTEPS%" "%DIR%\dsetup_%STAMP%.log" >nul
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

REM ---- PHASE 0c: the lever must have REACHED the model. A banner proves a value was
REM computed, never that it was used, so guard the CONSEQUENCE: the restored mixers are
REM +78,210 params, and a wrong count means RWKV_STRIP_CMIX did not reach the workers.
findstr /C:"Trainable parameters: 641862" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% WRONG_PARAM_COUNT -- the capacity lever did not reach the model %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_55 %DATE% %TIME% >> "%LOG%"
  exit /b 55
)
REM KD must be ABSENT here -- the opposite of every recent runner's guard.
findstr /C:"[kd-mix] KD ON" "%DIR%\decay_%STAMP%.log" >nul
if %ERRORLEVEL%==0 (
  echo %TAG% KD_LEAKED_IN %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_43 %DATE% %TIME% >> "%LOG%"
  exit /b 43
)
if not exist "C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ws10\w10gdry_d_%STEPS%.pth" (
  echo %TAG% DECAY_SHORT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_28 %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
echo %TAG% DECAY_OK %TIME% >> "%LOG%"

REM ---- PHASE C: rectified VAL-half eval, users 5001-7500 ----
set RWKV_EVAL_PAVA=1
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/cmixgraftdry w10gdry_d %DIR%\eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 5020 > "%DIR%\etoml_%STAMP%.log" 2>&1
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

REM Terminal marker BEFORE endlocal: endlocal restores the pre-setlocal environment,
REM so %LOG% would expand to empty and the marker would go to "".
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0