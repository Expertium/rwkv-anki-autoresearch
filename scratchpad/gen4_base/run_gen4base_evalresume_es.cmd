@echo off
REM ======================================================================================
REM gen4base EVAL RESUME attempt 4 with expandable segments (phase C only), sliced from run_gen4base_p2.cmd after the eval OOM'd at 04:50 on
REM 2026-09-03 with 1700/2500 users banked (user 6701, 2.4M rows; 36 GiB 'allocated' on a 12 GB card is
REM WDDM oversubscription after 1700 users of allocator churn). eval_sharded SKIPS completed users, so the
REM result jsonls are NOT deleted here and the run resumes at 6701 in a FRESH process. Markers go to
REM gen4base_evalresume.log. Original phase-2 header follows for provenance.
REM gen4base PHASE 2 (decay + eval), sliced from run_gen4base.cmd after the 2026-09-02 ~20:59 PC
REM restart killed the chain at decay step 10681 of 10935. The decay saves only on validate_iter
REM (VALIDATE_EVERY=100000), so nothing past step 50 survived; it re-runs from g4b_ws_10935.
REM Env block is BYTE-IDENTICAL to phase 1. Appends to the SAME log so the armed waiters see
REM DECAY_OK / DONE_EXIT_ exactly where they expect them.
REM
REM KD IS OFF IN BOTH ARMS, and not by preference: the teacher's features2card in_dim is
REM 92 while the gen-2 DBs feed 114, so it cannot forward-pass them. Comparing a KD-off
REM features run against the KD-ON champion would confound the features with KD removal,
REM worth ~0.0019 per iters 32/35/39/45.
REM
REM NEITHER ARM IS COMPARABLE TO iter 53. B minus A is the number that means something.
REM
REM Do NOT edit while running: cmd.exe re-reads a batch file from a saved byte offset.
REM ======================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base
set LOG=%DIR%\gen4base_evalresume.log
set STAMP=%RANDOM%%RANDOM%
set TAG=gen4base
set STEPS=10935

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

REM ---- what distinguishes this arm ----
set RWKV_ID_FEATURES=1
set RWKV_ZERO_FEATURES=
REM db overrides consumed by write_decay_setup.py / write_eval_toml.py, which used to
REM hardcode the 92-dim paths -- the trap the idfeat diagnostic caught.
set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_id4
set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_id4
set RWKV_LABEL_FILTER_DB=F:/rwkv_lmdb/label_filter_db_id_e2s
REM KD explicitly CLEARED, never merely unset, so an inherited value cannot turn the
REM control into a treatment -- the false-green shape that bit the rgate smoke.
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

if not exist "%DIR%" mkdir "%DIR%"
echo ===== gen4base EVAL RESUME START %DATE% %TIME% ===== >> "%LOG%"

if not exist "%DIR%\g4b_d_%STEPS%.pth" (
  echo %TAG% DECAY_CKPT_MISSING_FOR_RESUME %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_27 %DATE% %TIME% >> "%LOG%"
  exit /b 27
)

REM ---- PHASE C (RESUME): rectified VAL-half eval, users 5001-7500; completed users are skipped ----
set RWKV_EVAL_PAVA=1
REM attempt 4 (2026-09-03 12:05): user 6701 OOMs at the WDDM ceiling (36.09 GiB allocated + 5.9 GiB reserved
REM but unallocated, wanting 0.9 GiB more). Fragmentation is the immediate cause, so try expandable segments
REM once before excluding the user. The rule bank says this did not help the desktop-VRAM giants; different failure.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/gen4_base g4b_d %DIR%\eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% ETOML_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_24 %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM The eval toml must point at THIS arm's db, or the arm silently scores on the
REM other arm's feature width. This is trap #2 made into a guard.
findstr /C:"F:/rwkv_lmdb/test_db_5k_id4" "%DIR%\eval.toml" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_DB_MISMATCH %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_34 %DATE% %TIME% >> "%LOG%"
  exit /b 34
)
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
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