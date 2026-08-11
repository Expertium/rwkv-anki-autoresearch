@echo off
REM ===========================================================================================
REM BUDGET-CALIBRATION ARM RUNNER (2026-08-10). Called by run_cal.cmd as:
REM     call run_arm.cmd <TAG> <ARCH_MODULE> <INTERLEAVE 1|0>
REM TAG selects the toml (<TAG>_ws.toml), the ckpt prefixes and the result tags.
REM
REM WHY A SHARED ARM RUNNER: three arms differing in two env vars. Triplicating 40 env lines is
REM how iter 39 got launched into the wrong directory (a string-replace that missed one line).
REM
REM THE BUDGET CUT, and it takes TWO levers -- one per phase. ⚠ v1 OF THIS FILE GOT IT WRONG AND
REM THE RUN WAS KILLED 3.5 h IN (2026-08-10): it set only RWKV_MAX_STEPS, which caps the WS phase,
REM and assumed decay would follow. It does not. `write_decay_setup.py` derives decay length as
REM   total decay steps = int(decay_epochs * num_groups)
REM i.e. from EPOCHS x the dataset's group count (10,935 at MAX=65536), NOT from however many steps
REM WS actually ran. So a 3,645-step WS was followed by a 10,935-step decay: 14,580 total = 2/3
REM budget at a 1:3 WS:decay ratio -- a differently-SHAPED recipe, not a scaled-down one, which is
REM exactly what a calibration must not be.
REM   WS phase   -> RWKV_MAX_STEPS=%SHORTSTEPS%     (caps total_steps, train_rwkv.py:889)
REM   decay phase-> write_decay_setup arg 8 = %DECAYEPOCHS%  (fraction of an epoch)
REM ⚠ Do NOT shorten decay with RWKV_MAX_STEPS: decay is a COSINE to zero, so capping it would
REM stop the schedule mid-curve and leave the model at a high LR -- an unfinished decay, not a
REM short one. Shrinking decay_epochs keeps the cosine complete and just makes it shorter.
REM
REM Capping WS this way is safe because
REM   (a) WS LR is flat after warmup, so capping shortens the run rather than reshaping the
REM       schedule -- unlike the cosine phase above;
REM   (b) RWKV_KD_ALPHA is FIXED (0.9), so the teacher mix is undistorted -- with the ORIGINAL
REM       annealed 1->0 ramp, a shortened run would silently train at a different average alpha;
REM   (c) the end-of-training checkpoint is written at total_steps, so <TAG>_ws_3645.pth exists.
REM Result: WS 3,645 + decay 3,645 = 7,290 = exactly 1/3 of the champion's 21,870, at the SAME
REM 1:1 WS:decay proportion the champion uses.
REM EVAL STAYS AT THE FULL 2500 VAL-half users, deliberately: this experiment isolates the
REM TRAINING-budget axis. Cutting the eval set is a separate (variance-only, bias-free) change.
REM ===========================================================================================
setlocal
set TAG=%~1
set ARCHM=%~2
set ILV=%~3
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\budgetcal
set LOG=%DIR%\budgetcal.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set SHORTSTEPS=3645
REM 3645/10935 -- must match SHORTSTEPS/groups so decay equals WS, as at full budget.
set DECAYEPOCHS=0.3333

echo === ARM %TAG% START %DATE% %TIME% (arch=%ARCHM% interleave=%ILV% max_steps=%SHORTSTEPS%) === >> "%LOG%"

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=%ARCHM%
set RWKV_INTERLEAVE=%ILV%
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
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_KD_MIX=%DUMP%:10935
set RWKV_KD_ALPHA=0.9
REM ---- THE BUDGET CUT ----
set RWKV_MAX_STEPS=%SHORTSTEPS%
set RWKV_BENCH_WARMUP=100000

.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/budgetcal/%TAG%_ws.toml > "%DIR%\%TAG%_ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 21
)
if not exist "%DIR%\%TAG%_ws_%SHORTSTEPS%.pth" (
  echo ARM %TAG% NOCKPT %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 31
)
echo ARM %TAG% WS OK %TIME% >> "%LOG%"

set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_MAX_STEPS=

.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/budgetcal %TAG%_ws %TAG%_d scratchpad/budgetcal/%TAG%_decay.toml train_db_5k_h1 1 5000 %DECAYEPOCHS% 1e-3 65536 > "%DIR%\%TAG%_dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 22
)
REM Guard the v1 bug directly: the decay toml must carry the SHORT epoch count, not 1.0.
findstr /C:"EPOCHS = %DECAYEPOCHS%" "scratchpad\budgetcal\%TAG%_decay.toml" >nul
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% DECAYEPOCHS_NOT_APPLIED %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 34
)
findstr /C:"%TAG%_ws_%SHORTSTEPS%" "%DIR%\%TAG%_dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% WRONGCKPT %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 32
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/budgetcal/%TAG%_decay.toml > "%DIR%\%TAG%_decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 23
)
echo ARM %TAG% DECAY OK %TIME% >> "%LOG%"

.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/budgetcal %TAG%_d scratchpad/budgetcal/%TAG%_eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 7500 > "%DIR%\%TAG%_etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 24
)
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_EVAL_PAVA=1
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/budgetcal/%TAG%_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%TAG%_eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/budgetcal/%TAG%_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%TAG%_eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo ARM %TAG% EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 25
)
echo ARM %TAG% EVAL OK %TIME% >> "%LOG%"
endlocal & exit /b 0
