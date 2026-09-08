@echo off
REM w10plain -- ARM 1 OF THE ENDGAME: the plain 10+2 number. A 2-epoch decay branched off the
REM shared 10-epoch WS checkpoint (scratchpad/ws10/w10_ws_109350.pth), then the standard
REM rectified VAL-half eval. Andrew 2026-09-08: "keep current model size and do as you
REM suggested: train WS, then experiment with the decay stage."
REM
REM WHAT IT ANSWERS. realcyc -- the same recipe at 1+1 epochs -- scores ahead 0.298083 / imm
REM 0.263592. The 2026-08-11 budget calibration predicts a 10x budget is worth ~+0.0042 on ahead.
REM The stop criterion is ahead <= 0.2950, and the old d=128 model (2.76M params, ~12 epochs)
REM already reaches 0.294623 on this exact 2,499-user set. So this asks the question the phase
REM turns on: given the budget, does OUR 563k model land there too, or does capacity begin to
REM bind once the budget is real?
REM
REM It is also the REFERENCE every later decay-only branch (QAT, SAM, decay shape) is gated
REM against, so it runs FIRST and unmodified. Env is realcyc's, byte-identical; the only
REM differences from a normal iteration are the WS source and decay_epochs 1.0 -> 2.0 (21,870).
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\w10qat
set LOG=%DIR%\w10qat.log
set STAMP=%RANDOM%%RANDOM%
set TAG=w10qat
set STEPS=21870
set WSSTEPS=109350

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



REM ---- QAT: the ONLY difference from arm 1. Decay-only and warm-started, which is how QAT is
REM deployed here and what iter 40 (+0.0118 from scratch) says it must be.
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=reference/pq_cb_wkv_ws10_b10.txt
set RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_ws10_m2b12.txt
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ_LEARN=1

if not exist "%DIR%" mkdir "%DIR%"
echo ===== w10qat START %DATE% %TIME% ===== > "%LOG%"


REM ---- PHASE 0: the catalogs must EXIST and be the ws10-fitted ones, and the QAT env must reach
REM the FINAL config. A banner proves a value was computed, never that it was used: on 2026-08-12
REM the whole QAT env was parsed and then discarded by RWKV_ARCH_MODULE for an entire track-2
REM phase, symptomless except for a quantization cost of zero.
if not exist "reference\pq_cb_wkv_ws10_b10.txt" (
  echo %TAG% MISSING ws10 WKV catalog -- fit it from the ws10 WS-final first %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_51 %DATE% %TIME% >> "%LOG%"
  exit /b 51
)
if not exist "reference\pq_cb_shift_ws10_m2b12.txt" (
  echo %TAG% MISSING ws10 shift catalog %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_52 %DATE% %TIME% >> "%LOG%"
  exit /b 52
)
.venv\Scripts\python.exe scratchpad/qat_tax/assert_qat_live.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% QAT env is INERT in the final config %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_53 %DATE% %TIME% >> "%LOG%"
  exit /b 53
)

REM ---- PHASE B: the 2-EPOCH decay from the shared 10-epoch WS checkpoint ----
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/ws10 w10_ws w10q_d %DIR%\decay.toml F:/rwkv_lmdb/train_db_5k_h1_id5 1 5000 2.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
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
REM KD must be ABSENT here -- the opposite of every recent runner's guard.
findstr /C:"[kd-mix] KD ON" "%DIR%\decay_%STAMP%.log" >nul
if %ERRORLEVEL%==0 (
  echo %TAG% KD_LEAKED_IN %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_43 %DATE% %TIME% >> "%LOG%"
  exit /b 43
)
if not exist "%DIR%\w10q_d_%STEPS%.pth" (
  echo %TAG% DECAY_SHORT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_28 %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
echo %TAG% DECAY_OK %TIME% >> "%LOG%"

REM PHASE C is QUANT-AWARE here (the deploy number), so budget ~10 h, not the ~2.9 h a
REM plain eval takes -- iter 47's control eval ran 10h18m.
REM ---- PHASE C: rectified VAL-half eval, users 5001-7500 ----
set RWKV_EVAL_PAVA=1
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/w10qat w10q_d %DIR%\eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
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